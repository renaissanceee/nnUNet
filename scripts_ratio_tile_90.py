
# single-scale training and multi-scale testing setting proposed in mip-splatting

import os
import GPUtil
from concurrent.futures import ThreadPoolExecutor
import queue
import time
import os

losses = ['CELoss', 'DiceLoss', '']
factors = [1] * len(losses)

excluded_gpus = set([])
dry_run = False
jobs = list(zip(losses, factors))


def train_scene(gpu, loss, factor):
    gt_root = '/staging/leuven/stg_00081/jli/calibration/dataset/nnUNet_raw_nested/Dataset137_BraTS2021'
    pred_root = '/staging/leuven/stg_00081/jli/calibration/nnUNet_nested/nnUNet_results/Brats2021/Dataset137_BraTS2021'
    # ce_type = 'bins15'  # {'bins15', 'kde1'}
    tile = 90
    for fold in [0,1,2,3,4]:
        for ce_type in ['bins15', 'kde1']:
            # val
            gt_folder = f'{gt_root}/labelsVal_ece/fold_{fold}/'
            pred_folder = f'{pred_root}/nnUNetTrainer{loss}__nnUNetPlans__2d/fold_{fold}/validation_ece'
            cmd = f"nnUNetv2_ratio_estimator --biomarker=ntr --ece_percentage={tile} --ce_type={ce_type} {gt_folder} {pred_folder}"
            print(cmd)
            if not dry_run:
                os.system(cmd)
            # test
            cmd = f"nnUNetv2_ratio_estimator --biomarker=ntr --ece_percentage={tile} --ce_type={ce_type} {gt_folder.replace('labelsVal_ece', 'labelsTs')} {pred_folder.replace('validation_ece','test')}"
            print(cmd)
            if not dry_run:
                os.system(cmd)

            # #---------------- after_TS #----------------
            # # val+TS
            # gt_folder = f'{gt_root}/labelsVal_ece/fold_{fold}/'
            # pred_folder = f'{pred_root}/nnUNetTrainer{loss}__nnUNetPlans__2d/fold_{fold}/validation_ece'
            # cmd = f"nnUNetv2_ratio_estimator --biomarker=ntr --ece_percentage={tile} --TS list_1000_new --ce_type={ce_type} {gt_folder} {pred_folder}"
            # print(cmd)
            # if not dry_run:
            #     os.system(cmd)
            # # test+TS
            # cmd = f"nnUNetv2_ratio_estimator --biomarker=ntr --ece_percentage={tile} --TS list_1000_new --ce_type={ce_type} {gt_folder.replace('labelsVal_ece', 'labelsTs')} {pred_folder.replace('validation_ece','test')}"
            # print(cmd)
            # if not dry_run:
            #     os.system(cmd)

    return True


def worker(gpu, loss, factor):
    print(f"Starting job: loss {loss}\n")
    train_scene(gpu, loss, factor)
    print(f"Finished job: loss {loss}\n")
    # This worker function starts a job and returns when it's done.


def dispatch_jobs(jobs, executor):
    future_to_job = {}
    reserved_gpus = set()  # GPUs that are slated for work but may not be active yet

    while jobs or future_to_job:
        # Get the list of available GPUs, not including those that are reserved.
        all_available_gpus = set(GPUtil.getAvailable(order="first", limit=10, maxMemory=0.1))
        available_gpus = list(all_available_gpus - reserved_gpus - excluded_gpus)

        # Launch new jobs on available GPUs
        while available_gpus and jobs:
            gpu = available_gpus.pop(0)
            job = jobs.pop(0)
            future = executor.submit(worker, gpu, *job)  # Unpacking job as arguments to worker
            future_to_job[future] = (gpu, job)

            reserved_gpus.add(gpu)  # Reserve this GPU until the job starts processing

        # Check for completed jobs and remove them from the list of running jobs.
        # Also, release the GPUs they were using.
        done_futures = [future for future in future_to_job if future.done()]
        for future in done_futures:
            job = future_to_job.pop(future)  # Remove the job associated with the completed future
            gpu = job[0]  # The GPU is the first element in each job tuple
            reserved_gpus.discard(gpu)  # Release this GPU
            print(f"Job {job} has finished., rellasing GPU {gpu}")
        # (Optional) You might want to introduce a small delay here to prevent this loop from spinning very fast
        # when there are no GPUs available.
        time.sleep(5)

    print("All jobs have been processed.")


# Using ThreadPoolExecutor to manage the thread pool
with ThreadPoolExecutor(max_workers=8) as executor:
    dispatch_jobs(jobs, executor)
