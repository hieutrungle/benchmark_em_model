import torch
import torchvision
import models.efficientnet as efficientnet
import data_io
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import time
import timer
import logger

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def prediction(conductivities):
    weight_path = "/home/hieule/research/benchmark_em_model/saved_models/061/model_20240208_171425_0.pt"
    model = efficientnet.efficientnet_prediction_model(num_classes=1)
    # load weight
    checkpoint = torch.load(weight_path, map_location=DEVICE)
    model.load_state_dict(checkpoint)

    # TODO: get prediction data
    test_dir = "/home/hieule/research/benchmark_em_model/data/061/test"
    test_ds = data_io.ImageCurrentDataset(
        test_dir,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        ),
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )
    conductivity_ds = data_io.DatasetWrapper(conductivities)
    conductivity_loader = torch.utils.data.DataLoader(
        conductivity_ds,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )
    # TODO: forward test data using loaded model
    model.eval()
    all_pred_currents = []
    with timer.Timer(logger_fn=logger.log):
        with torch.no_grad():
            for i, ((images, currents), (conductivity)) in enumerate(
                zip(test_loader, conductivity_loader)
            ):
                # images: [B, C, H, W]
                # currents: [B, 1]
                pred_currents = model.forward(images)
                pred_currents = pred_currents.detach().numpy()
                # pred_currents = pred_currents * (1/conductivity.numpy())
                all_pred_currents.extend(list(pred_currents[:, 0]))

    all_pred_currents = np.array(all_pred_currents)

    # for i in range(len(all_pred_currents)):
    #     if all_pred_currents[i] < 0.005:
    #         all_pred_currents[i] = 0

    # scaler = preprocessing.MinMaxScaler()
    # scaler = scaler.fit(all_pred_currents)
    # all_pred_currents =  scaler.inverse_transform(all_pred_currents)
    # all_pred_currents = all_pred_currents*13.61189 + 2.474435

    # for (conductivity, pred) in zip(conductivities, all_pred_currents):
    #  print(conductivity, pred)

    print(f"all_pred_currents: {all_pred_currents.shape}")
    # all_pred_resistance = voltage/all_pred_currents
    # all_pred_resistance = all_pred_resistance * (1.0 / np.array(conductivities))
    all_pred_currents = all_pred_currents * np.array(conductivities)
    print(all_pred_currents)
    # to excel
    df = pd.DataFrame(all_pred_currents)
    pd.DataFrame.to_excel(df, "./prediction.xlsx")


if __name__ == "__main__":
    conductivities = [1] * 175
    voltage = 1.0
    prediction(conductivities)
