import torch
import pandas as pd
from tqdm import tqdm
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def extract_features(model, train_loader):

    features1 = []
    features2 = []
    features3 = []
    features4 = []
    features5 = []
    features6 = []
    features7 = []
    features8 = []
    features9 = []
    features10 = []
    labels_p = []

    model.eval()

    with torch.no_grad():

        for inputs,targets in tqdm(train_loader):

            inputs = inputs.to(device)
            targets = targets.to(device)

            x1 = model.layer1(inputs)
            x11 = x1.view(x1.size(0), x1.size(1), -1)
            features1.extend(torch.mean(x11,2))

            x2 = model.layer2(x1)
            x22 = x2.view(x2.size(0), x2.size(1), -1)
            features2.extend(torch.mean(x22,2))

            x3 = model.layer3(x2)
            x33 = x3.view(x3.size(0), x3.size(1), -1)
            features3.extend(torch.mean(x33,2))

            x4 = model.layer4(x3)
            x44 = x4.view(x4.size(0), x4.size(1), -1)
            features4.extend(torch.mean(x44,2))

            x5 = model.layer5(x4)
            x55 = x5.view(x5.size(0), x5.size(1), -1)
            features5.extend(torch.mean(x55,2))

            x6 = model.layer6(x5)
            x66 = x6.view(x6.size(0), x6.size(1), -1)
            features6.extend(torch.mean(x66,2))

            x7 = model.layer7(x6)
            x77 = x7.view(x7.size(0), x7.size(1), -1)
            features7.extend(torch.mean(x77,2))

            x8 = model.layer8(x7)
            x88 = x8.view(x8.size(0), x8.size(1), -1)
            features8.extend(torch.mean(x88,2))

            x9 = model.layer9(x8)
            x99 = x9.view(x9.size(0), x9.size(1), -1)
            features9.extend(torch.mean(x99,2))

            x10 = model.layer10(x9)
            x1010 = x10.view(x10.size(0), x10.size(1), -1)
            features10.extend(torch.mean(x1010,2))

            labels_p.extend(targets)

    ############################################
    # Stack tensors
    ############################################

    features1 = torch.stack(features1)
    features2 = torch.stack(features2)
    features3 = torch.stack(features3)
    features4 = torch.stack(features4)
    features5 = torch.stack(features5)
    features6 = torch.stack(features6)
    features7 = torch.stack(features7)
    features8 = torch.stack(features8)
    features9 = torch.stack(features9)
    features10 = torch.stack(features10)
    labels_p = torch.stack(labels_p)

    ############################################
    # Save CSV
    ############################################

    os.makedirs("outputs/features", exist_ok=True)

    pd.DataFrame(features1.cpu().numpy()).to_csv("outputs/features/L1_pn.csv", index=False)
    pd.DataFrame(features2.cpu().numpy()).to_csv("outputs/features/L2_pn.csv", index=False)
    pd.DataFrame(features3.cpu().numpy()).to_csv("outputs/features/L3_pn.csv", index=False)
    pd.DataFrame(features4.cpu().numpy()).to_csv("outputs/features/L4_pn.csv", index=False)
    pd.DataFrame(features5.cpu().numpy()).to_csv("outputs/features/L5_pn.csv", index=False)
    pd.DataFrame(features6.cpu().numpy()).to_csv("outputs/features/L6_pn.csv", index=False)
    pd.DataFrame(features7.cpu().numpy()).to_csv("outputs/features/L7_pn.csv", index=False)
    pd.DataFrame(features8.cpu().numpy()).to_csv("outputs/features/L8_pn.csv", index=False)
    pd.DataFrame(features9.cpu().numpy()).to_csv("outputs/features/L9_pn.csv", index=False)
    pd.DataFrame(features10.cpu().numpy()).to_csv("outputs/features/L10_pn.csv", index=False)
    pd.DataFrame(labels_p.cpu().numpy()).to_csv("outputs/features/labels_pn.csv", index=False)

    print("Feature extraction completed.")