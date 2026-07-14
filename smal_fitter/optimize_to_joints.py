import os

# CUDA_VISIBLE_DEVICES must be set before `import torch` (torch >= 2.3 raises
# an INTERNAL ASSERT if CVD changes after CUDA init). Guarded on __main__ so
# importing this module as a library has no CVD side-effect on the importer.
# See issue #73.
if __name__ == "__main__":
    import config as _cfg_for_env

    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", _cfg_for_env.GPU_IDS)
    del _cfg_for_env

import numpy as np
import argparse

from smal_fitter.fitter import SMALFitter
import pickle as pkl

import torch
import imageio

from smal_fitter.data_loader import load_badja_sequence, load_stanford_sequence, load_SMIL_sequence
import trimesh

from tqdm import trange


class ImageExporter:
    def __init__(self, output_dir, filenames):
        self.output_dirs = self.generate_output_folders(output_dir, filenames)
        self.stage_id = 0
        self.epoch_name = 0

    def generate_output_folders(self, root_directory, filename_batch):
        if not os.path.exists(root_directory):
            os.mkdir(root_directory)

        output_dirs = []
        for filename in filename_batch:
            filename_path = os.path.join(root_directory, os.path.splitext(filename)[0])
            output_dirs.append(filename_path)
            if not os.path.exists(filename_path):
                os.mkdir(filename_path)

        return output_dirs

    def export(self, collage_np, batch_id, global_id, img_parameters, vertices, faces, img_idx=0, epoch=None):
        epoch_name = epoch if epoch is not None else self.epoch_name
        imageio.imsave(
            os.path.join(self.output_dirs[global_id], "st{0}_ep{1}.png".format(self.stage_id, epoch_name)), collage_np
        )

        # Export parameters
        with open(
            os.path.join(self.output_dirs[global_id], "st{0}_ep{1}.pkl".format(self.stage_id, epoch_name)), "wb"
        ) as f:
            pkl.dump(img_parameters, f)

        # Export mesh
        vertices = vertices[batch_id].cpu().numpy()
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        mesh.export(os.path.join(self.output_dirs[global_id], "st{0}_ep{1}.ply".format(self.stage_id, epoch_name)))


def main():

    parser = argparse.ArgumentParser(description="SMAL Fitter")
    parser.add_argument("--test", action="store_true", help="Run in testing mode")
    args = parser.parse_args()

    if args.test:
        from tests import config_test as config
    else:
        import config
    # CUDA_VISIBLE_DEVICES is set at the top of this file, before torch is imported.

    if not os.path.exists(config.OUTPUT_DIR):
        os.makedirs(config.OUTPUT_DIR)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset, name = config.SEQUENCE_OR_IMAGE_NAME.split(":")

    if dataset == "badja":
        data, filenames = load_badja_sequence(config.BADJA_PATH, name, config.CROP_SIZE, image_range=config.IMAGE_RANGE)
    elif dataset == "stanfordextra":
        data, filenames = load_stanford_sequence(config.STANFORD_EXTRA_PATH, name, config.CROP_SIZE)
    else:
        data, filenames = load_SMIL_sequence(config.REPLICANT_PATH, name, config.CROP_SIZE)

    dataset_size = len(filenames)
    print("Dataset size: {0}".format(dataset_size))

    if not config.ignore_hardcoded_body:
        assert config.SHAPE_FAMILY >= 0, "Shape family should be greater than 0"

        use_unity_prior = config.SHAPE_FAMILY == 1 and not config.FORCE_SMAL_PRIOR
    else:
        use_unity_prior = False

    if not use_unity_prior and not config.ALLOW_LIMB_SCALING:
        print(
            "WARNING: Limb scaling is only recommended for the new Unity prior. TODO: add a regularizer to constrain scale parameters."
        )
        config.ALLOW_LIMB_SCALING = False

    image_exporter = ImageExporter(config.OUTPUT_DIR, filenames)

    model = SMALFitter(device, data, config.WINDOW_SIZE, config.SHAPE_FAMILY, use_unity_prior)
    for stage_id, weights in enumerate(np.array(config.OPT_WEIGHTS).T):
        opt_weight = weights[:6]
        w_temp = weights[6]
        epochs = int(weights[7])
        lr = weights[8]

        optimizer = torch.optim.Adam(
            [
                {
                    "params": [param for name, param in model.named_parameters() if name != "fov"],
                    "lr": lr,
                },  # Exclude fov
                {"params": [model.fov], "lr": 1},  # Include fov with its own (much higher) learning rate
            ],
            lr=lr,
            betas=(0.5, 0.999),
        )

        if stage_id == 0:
            model.joint_rotations.requires_grad = False
            model.betas.requires_grad = False
            model.log_beta_scales.requires_grad = False
            model.fov.requires_grad = True
            target_visibility = model.target_visibility.clone()
            model.target_visibility *= 0
            model.target_visibility[:, config.TORSO_JOINTS] = target_visibility[
                :, config.TORSO_JOINTS
            ]  # Turn on only torso points
        else:
            model.joint_rotations.requires_grad = True
            model.betas.requires_grad = True
            model.fov.requires_grad = True
            if config.ALLOW_LIMB_SCALING:
                model.log_beta_scales.requires_grad = True
            model.target_visibility = data[-1].clone()

        t = trange(epochs, leave=True)
        for epoch_id in t:
            image_exporter.stage_id = stage_id
            image_exporter.epoch_name = str(epoch_id)

            acc_loss = 0
            optimizer.zero_grad()
            for j in range(0, dataset_size, config.WINDOW_SIZE):
                batch_range = list(range(j, min(dataset_size, j + config.WINDOW_SIZE)))
                loss, losses = model(batch_range, opt_weight, stage_id)
                acc_loss += loss.mean()
                # print ("Optimizing Stage: {}\t Epoch: {}, Range: {}, Loss: {}, Detail: {}".format(stage_id, epoch_id, batch_range, loss.data, losses))

            # get weighted losses from model (the weights refer to those set up in the config)
            joint_loss, global_loss, trans_loss = model.get_temporal(w_temp)

            desc = "EPOCH: Optimizing Stage: {}\t Epoch: {}, Loss: {:.2f}, Temporal: ({}, {}, {})".format(
                stage_id, epoch_id, acc_loss.data, joint_loss.data, global_loss.data, trans_loss.data
            )

            t.set_description(desc)
            t.refresh()

            # get loss from all partial losses
            acc_loss = acc_loss + joint_loss + global_loss + trans_loss
            # get gradients by running backprop
            acc_loss.backward()
            # update parameters
            optimizer.step()

            if epoch_id % config.VIS_FREQUENCY == 0:
                model.generate_visualization(image_exporter)

    image_exporter.stage_id = 10
    image_exporter.epoch_name = str(0)
    model.generate_visualization(image_exporter)  # Final stage


if __name__ == "__main__":
    # Set CVD before torch touches CUDA. Safe here because __main__ runs after
    # imports, but torch.cuda has not been initialized yet in typical use, any
    # entrypoint that needs strict pre-torch CVD should set it in its own
    # top-of-file block (see train_smil_regressor.py, benchmark_model.py).
    import config as _cfg_for_env

    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", _cfg_for_env.GPU_IDS)
    del _cfg_for_env
    main()
