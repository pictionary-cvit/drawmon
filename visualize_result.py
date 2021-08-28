import sys
from matplotlib import pyplot as plt
from data_pictText import ImageInputGenerator
import argparse
import json

parser = argparse.ArgumentParser("visualise")
parser.add_argument("--data-path", type=str, dest="data_path", required=True)
parser.add_argument("--batch-size", type=int, dest="batch_size", default=1)
parser.add_argument("--output-path", type=str, dest="output_path", default="./renders")
args = parser.parse_args()

# model = TBPP512_dense_separable(
#     input_shape=(512, 512, 1),
#     softmax=True,
#     scale=0.9,
#     isQuads=False,
#     isRbb=False,
#     num_dense_segs=3,
#     use_prev_feature_map=False,
#     num_multi_scale_maps=5,
#     num_classes=5,
#     activation="tfa_mish",
# )

gen_val = ImageInputGenerator(
    args.data_path, args.batch_size, "val", give_idx=False
).get_dataset()

for i, item in enumerate(gen_val):
    plt.imshow(item[0].numpy()[0, :, :, 0])
    plt.title(item[1].numpy())
    plt.savefig(f"./{args.output_path}/{i}.png")
    plt.close()

    with open(f"./{args.output_path}/{i}.txt", "w") as fil:
        json.dump(item[1].numpy(), fil)
