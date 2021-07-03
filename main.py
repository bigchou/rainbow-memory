import logging.config
import os
import random
from collections import defaultdict

import numpy as np
import torch, rmcifar, pdb, json, rmtinyimagenet, rmdog
from copy import deepcopy
from randaugment import RandAugment
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from configuration import config
from utils.augment import Cutout, select_autoaugment
from utils.data_loader import get_statistics#,get_test_datalist
from utils.data_loader import get_train_datalist
from utils.method_manager import select_method
from utils.timer import central_timer
from utils.data_loader import get_train_collection_name
from evaluate import eval, compute_acc

def main():
    args = config.base_parser()

    # Save file name
    tr_names = ""
    for trans in args.transforms:
        tr_names += "_" + trans
    now = central_timer
    save_path = f"{now}_{args.dataset}/{args.mode}_{args.mem_manage}_{args.stream_env}_msz{args.memory_size}_rnd{args.rnd_seed}{tr_names}"
    os.makedirs(f"logs/{now}_{args.dataset}", exist_ok=True)#create logs
    os.makedirs(f"results/{now}_{args.dataset}", exist_ok=True)#create results
    writer = SummaryWriter(f"tensorboard/{central_timer}")#create tensorboard
    args.save_path = f"results/{now}_{args.dataset}"#save model path <------------



    # Set logger
    logging.config.fileConfig("./configuration/logging.conf")
    logger = logging.getLogger()
    fileHandler = logging.FileHandler("logs/{}.log".format(save_path), mode="w")
    formatter = logging.Formatter(
        "[%(levelname)s] %(filename)s:%(lineno)d > %(message)s"
    )
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)

    # Set device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info(f"Set the device ({device})")

    # Fix the random seeds
    # https://hoya012.github.io/blog/reproducible_pytorch/
    torch.manual_seed(args.rnd_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.rnd_seed)
    random.seed(args.rnd_seed)

    # Transform Definition
    train_transform = []
    if "cutout" in args.transforms:
        train_transform.append(Cutout(size=16))
    if "randaug" in args.transforms:
        train_transform.append(RandAugment())
    if "autoaug" in args.transforms:
        train_transform.append(select_autoaugment(args.dataset))

    if "dog" in args.dataset:#dog120
        crop_im_size = 224
        n_classes = 120
        mean, std, _, _, _ = get_statistics(dataset='imagenet1000')
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=crop_im_size),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        logger.info(f"Using train-transforms {train_transform}")
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(crop_im_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:#TinyImagenet,cifar100
        mean, std, n_classes, inp_size, _ = get_statistics(dataset=args.dataset)
        train_transform = transforms.Compose(
            [
                transforms.Resize((inp_size, inp_size)),
                transforms.RandomCrop(inp_size, padding=4),
                transforms.RandomHorizontalFlip(),
                *train_transform,
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        logger.info(f"Using train-transforms {train_transform}")

        test_transform = transforms.Compose(
            [
                transforms.Resize((inp_size, inp_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

    logger.info(f"[1] Select a CIL method ({args.mode})")
    criterion = nn.CrossEntropyLoss(reduction="mean")
    method = select_method(
        args, criterion, device, train_transform, test_transform, n_classes
    )

    logger.info(f"[2] Incrementally training {args.n_tasks} tasks")
    task_records = defaultdict(list)

    ###############################
    prev_gallery_features, prev_gallery_labels, prev_gallery_names = None, None, [] # for retrieval setting
    ###############################

    for cur_iter in range(args.n_tasks):
        if args.mode == "joint" and cur_iter > 0:
            return

        print("\n" + "#" * 50)
        print(f"# Task {cur_iter} iteration")
        print("#" * 50 + "\n")
        logger.info("[2-1] Prepare a datalist for the current task")

        task_acc = 0.0
        eval_dict = dict()

        #*********************************************************************************************************
        # get datalist
        #cur_train_datalist = get_train_datalist(args, cur_iter)
        #cur_test_datalist = get_test_datalist(args, args.exp_name, cur_iter)
        
        np.random.seed(args.rnd_seed)
        cur_train_datalist, cur_test_datalist = [], []
        for cur_iter_ in range(args.n_tasks):
            colname = get_train_collection_name(dataset=args.dataset,exp=args.exp_name,rnd=args.rnd_seed,n_cls=args.n_cls_a_task,iter=cur_iter_,)
            if args.dataset == "TinyImagenet":
                with open(f"collections/tinyimagenet200/{colname}.json","r") as f: X = json.load(f)
                test_size = 0.05
            else:#dog120, cifar100
                with open(f"collections/{args.dataset}/{colname}.json","r") as f: X = json.load(f)
                test_size = 0.1
            y = [item['label'] for item in X]
            X_train, X_val, _, _ = train_test_split(X, y,stratify=y, test_size=test_size,random_state=args.rnd_seed)# seed for reproducibility
            if cur_iter == cur_iter_:
                cur_train_datalist.extend(X_train)
                logger.info(f"[Train] Get datalist {cur_iter_} from {colname}.json")
            if "blur" in args.exp_name:
                cur_test_datalist.extend(X_val)
                logger.info(f"[Test] Get datalist {cur_iter_} from {colname}.json")
            else:#"general","disjoint"
                if cur_iter_ <= cur_iter:
                    cur_test_datalist.extend(X_val)
                    logger.info(f"[Test] Get datalist {cur_iter_} from {colname}.json")
        #*********************************************************************************************************
        


        logger.info("[2-2] Set environment for the current task")
        method.set_current_dataset(cur_train_datalist, cur_test_datalist)
        # Increment known class for current task iteration.
        method.before_task(cur_train_datalist, cur_iter, args.init_model, args.init_opt)

        # The way to handle streamed samles
        logger.info(f"[2-3] Start to train under {args.stream_env}")


        ##############################################
        if args.mode == "ewc":# original ewc stores no memory
            method.memory_list = []
        ##############################################


        if args.stream_env == "offline" or args.mode == "joint" or args.mode == "gdumb":
            # Offline Train
            task_acc, eval_dict = method.train(
                cur_iter=cur_iter,
                n_epoch=args.n_epoch,
                batch_size=args.batchsize,
                n_worker=args.n_worker,
            )
            if args.mode == "joint":
                logger.info(f"joint accuracy: {task_acc}")
        """
        elif args.stream_env == "online":
            # Online Train
            logger.info("Train over streamed data once")
            method.train(cur_iter=cur_iter,n_epoch=1,batch_size=args.batchsize,n_worker=args.n_worker,)
            method.update_memory(cur_iter)
            # No stremed training data, train with only memory_list
            method.set_current_dataset([], cur_test_datalist)
            logger.info("Train over memory")
            task_acc, eval_dict = method.train(cur_iter=cur_iter,n_epoch=args.n_epoch,batch_size=args.batchsize,n_worker=args.n_worker,)
            method.after_task(cur_iter)
        """

        logger.info("[2-4] Update the information for the current task")
        method.after_task(cur_iter)
        task_records["task_acc"].append(task_acc)
        # task_records['cls_acc'][k][j] = break down j-class accuracy from 'task_acc'
        task_records["cls_acc"].append(eval_dict["cls_acc"])

        # Notify to NSML
        logger.info("[2-5] Report task result")
        writer.add_scalar("Metrics/TaskAcc", task_acc, cur_iter)










        #pdb.set_trace()
        ###########################################################################################
        # Evaluate Image Retrieval
        if "cifar" in args.dataset:
            rmdataset = rmcifar.CIFAR100_BLUR10 if "blur" in args.exp_name else rmcifar.CIFAR100_DISJOINT
        elif "imagenet" in args.dataset.lower():#To match TinyImagenet
            rmdataset = rmtinyimagenet.TINYIMAGENET_BLUR30 if "blur" in args.exp_name else rmtinyimagenet.TINYIMAGENET_DISJOINT
        elif "dog" in args.dataset:
            rmdataset = rmdog.DOG_DISJOINT
        else:
            raise NotImplementedError("Implementation Not Found")
        ### load gallery data
        if "blur" in args.exp_name:
            eval_trainset = rmdataset(mode="gallery", session_id=cur_iter,seed=args.rnd_seed)
        else:
            eval_trainset = rmdataset(mode="gallery", session_id=cur_iter,seed=args.rnd_seed,exp_name=args.exp_name)
        eval_trainloader = DataLoader(eval_trainset, batch_size=100, shuffle=False, num_workers=8)
        ### load testing query data
        if "blur" in args.exp_name:
            testset = rmdataset(mode="test",session_id=cur_iter,seed=args.rnd_seed)
        else:
            testset = rmdataset(mode="test",session_id=cur_iter,seed=args.rnd_seed,exp_name=args.exp_name)
        testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)
        # load model with highest validation accuracy
        ckptpath = '%s/ckpt_session%d.pth'%(method.save_path,cur_iter)
        ckpt = torch.load(ckptpath)
        tmpnet = deepcopy(method.model)
        tmpnet.load_state_dict(ckpt['net'])
        print("load model with highest validation accuracy")
        record, curr_gallery_features, curr_gallery_labels, curr_gallery_names = eval(
            tmpnet, testloader, eval_trainloader, 8,
            session_id=cur_iter,
            reindex=False,
            prev_gallery_features=prev_gallery_features,
            prev_gallery_labels=prev_gallery_labels,
            return_curr_gallery_names = True,
        )
        final_acc = compute_acc(tmpnet,testloader)
        record["cls_acc"] = final_acc
        
        if prev_gallery_features is not None:
            prev_gallery_features = np.vstack((curr_gallery_features,prev_gallery_features))
            prev_gallery_labels = np.vstack((curr_gallery_labels,prev_gallery_labels))
            prev_gallery_names = curr_gallery_names + prev_gallery_names
        else:#prev_gallery_features is None
            prev_gallery_features = curr_gallery_features
            prev_gallery_labels = curr_gallery_labels
            prev_gallery_names = curr_gallery_names
        
        np.save('%s/gallery_features_session%d.npy'%(method.save_path,cur_iter),prev_gallery_features)
        np.save('%s/gallery_labels_session%d.npy'%(method.save_path,cur_iter),prev_gallery_labels)
        np.save('%s/gallery_names_session%d.npy'%(method.save_path,cur_iter),prev_gallery_names)
        with open(os.path.join(method.save_path,"result_session%d.json"%(cur_iter)),"w",encoding="utf-8") as f: json.dump(record,f)


    np.save(f"results/{save_path}.npy", task_records["task_acc"])

    # Accuracy (A)
    A_avg = np.mean(task_records["task_acc"])
    A_last = task_records["task_acc"][args.n_tasks - 1]

    # Forgetting (F)
    acc_arr = np.array(task_records["cls_acc"])
    # cls_acc = (k, j), acc for j at k
    cls_acc = acc_arr.reshape(-1, args.n_cls_a_task).mean(1).reshape(args.n_tasks, -1)
    for k in range(args.n_tasks):
        forget_k = []
        for j in range(args.n_tasks):
            if j < k:
                forget_k.append(cls_acc[:k, j].max() - cls_acc[k, j])
            else:
                forget_k.append(None)
        task_records["forget"].append(forget_k)
    F_last = np.mean(task_records["forget"][-1][:-1])

    # Intrasigence (I)
    I_last = args.joint_acc - A_last

    logger.info(f"======== Summary =======")
    logger.info(f"A_last {A_last} | A_avg {A_avg} | F_last {F_last} | I_last {I_last}")


if __name__ == "__main__":
    main()
