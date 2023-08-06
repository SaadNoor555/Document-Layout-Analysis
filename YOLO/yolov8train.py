import os
from tqdm import tqdm

def all_files_in_folder_symlink(source_dir, target_dir):
    files = os.listdir(source_dir)

    for file in tqdm(files):
        source_file = os.path.join(source_dir, file)
        target_file = os.path.join(target_dir, file)
        os.symlink(source_file, target_file)

# all_files_in_folder_symlink("/media/abhijit/New Volume/dls2/data/badlad/labels/yolov8_format/train","/datasets/badlad/labels/train") 
# all_files_in_folder_symlink("/media/abhijit/New Volume/dls2/data/badlad/images/train","/datasets/badlad/images/train")
# all_files_in_folder_symlink("/media/abhijit/New Volume/dls2/data/badlad/images/test","/datasets/badlad/images/test")

file_content = """
path: /media/abhijit/New Volume/dls2/code/datasets/badlad/heq  # dataset root dir
train: images/train  # train images (relative to 'path') 128 images
val: images/val  # val images (relative to 'path') 128 images
test:  images/test

# Classes
names:
  0: paragraph
  1: text_box
  2: image
  3: table


"""

with open("yolov8.yaml", mode="w") as f:
    f.write(file_content)



# from ultralytics import YOLO
# model = YOLO("runs/segment/train17/weights/best_43.pt")
# # help(YOLO)

# model.train(data="yolov8.yaml", epochs=7, device=[0],imgsz=672, overlap_mask=True, mask_ratio=3, lr0=0.008, lrf=0.001)

from ultralytics import YOLO
from ray import tune
import wandb
# wandb.init(mode="disabled")

# model = YOLO("yolov8m-seg.pt")

# # Run Ray Tune on the model
# result_grid = model.tune(data="yolov8.yaml",
#                          space={"lr0": tune.uniform(1e-5, 1e-1),
#                                 "lrf": tune.uniform(0.01, 1.0),
#                                 "momentum": tune.uniform(0.6, 0.98) ,
#                                 "weight_decay": tune.uniform(0.0, 0.001),
#                                 "box": tune.uniform(0.02, 0.2),
#                                 "cls": tune.uniform(0.2, 4.0),
#                                 "hsv_h": tune.uniform(0.0, 0.1),
#                                 "hsv_s": tune.uniform(0.0, 0.9),
#                                 "hsv_v": tune.uniform(0.0, 0.9),
#                                 },
#                          epochs=50)

# if result_grid.errors:
#     print("One or more trials failed!")
# else:
#     print("No errors!")


# for i, result in enumerate(result_grid):
#     print(f"Trial #{i}: Configuration: {result.config}, Last Reported Metrics: {result.metrics}")


# import matplotlib.pyplot as plt

# for result in result_grid:
#     plt.plot(result.metrics_dataframe["training_iteration"], result.metrics_dataframe["mean_accuracy"], label=f"Trial {i}")

# plt.xlabel('Training Iterations')
# plt.ylabel('Mean Accuracy')
# plt.legend()
# plt.show()


import os
import yoloRay
from yoloRay import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch

model = YOLO("yolov8m-seg.pt")

config = {
        "lr0": tune.uniform(1e-5, 1e-1),
        "lrf": tune.uniform(0.01, 1.0),
        "momentum": tune.uniform(0.6, 0.98) ,
        "weight_decay": tune.uniform(0.0, 0.001),
        "box": tune.uniform(0.02, 0.2),
        "cls": tune.uniform(0.2, 4.0),
        "hsv_h": tune.uniform(0.0, 0.1),
        "hsv_s": tune.uniform(0.0, 0.9),
        "hsv_v": tune.uniform(0.0, 0.9),
        "perspective": tune.uniform(0.0, 0.001),
    }
scheduler = ASHAScheduler(
    # max_t=2,
    grace_period=10,
    reduction_factor=2)

tuner = tune.Tuner(
    tune.with_resources(
        tune.with_parameters(model.train(data="yolov8.yaml", epochs=150, device=[0], imgsz=672, overlap_mask=True, mask_ratio=3, save_period=50, patience=50)),
        resources={"cpu": 10, "gpu": 1}
    ),
    tune_config=tune.TuneConfig(
        metric="loss",
        mode="min",
        scheduler=scheduler,
        num_samples=7,
    ),
    param_space=config,
)
results = tuner.fit()

best_result = results.get_best_result("loss", "min")

print("Best trial config: {}".format(best_result.config))
print("Best trial final validation loss: {}".format(
    best_result.metrics["loss"]))
print("Best trial final validation accuracy: {}".format(
    best_result.metrics["accuracy"]))


