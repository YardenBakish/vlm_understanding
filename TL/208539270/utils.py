import gzip
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import re
import matplotlib.pyplot as plt

class TMLDataset(Dataset):
    def __init__(self, part, fpath='dataset-full.npz', transform=None):
        # init
        with gzip.open(fpath, 'rb') as fin:
            data_tr = np.load(fin, allow_pickle=True)
            data_test = np.load(fin, allow_pickle=True)
        if part=='train':
            self.data = data_tr
        elif part=='test':
            self.data = data_test
        else:
            raise ValueError(f'Unknown dataset part {part}')
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        if self.transform:
            x = self.transform(x)
        return x, y
   
def standard_train(model, data_tr, criterion, optimizer, lr_scheduler, device,
                   epochs=100, batch_size=128, dl_nw=10, reg_strength=0.0):
    loader_tr = DataLoader(data_tr,
                           batch_size=batch_size,
                           shuffle=True,
                           pin_memory=True,
                           num_workers=4)

    for epoch in range(epochs):
        for i, data in enumerate(loader_tr, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()

            pred = model(inputs)
            loss = criterion(pred, labels)

            # add sparsity regularization if requested
            if reg_strength > 0.0 and model._activation is not None:
                loss = loss + reg_strength * model._activation.abs().mean()

            loss.backward()
            optimizer.step()

        lr_scheduler.step()

    return model



def compute_accuracy(model, data_loader, device):
    count_correct = 0
    count_all = 0
    with torch.no_grad():
        for data in data_loader:
            x, y = data[0].to(device), data[1].to(device)
            outputs = model(x)
            _, preds = torch.max(outputs, 1)
            count_correct += torch.sum(y==preds).to('cpu')
            count_all += len(x)
    return count_correct/float(count_all)

def compute_backdoor_success_rate(model, data_loader, device,
                                  mask, trigger, c_t):
    count_success = 0
    count_all = 0
    with torch.no_grad():
        for data in data_loader:
            x, y = data[0], data[1]
            x = x[y!=c_t]
            if len(x)<1:
                continue
            x = data[0].to(device)
            x = x*(1-mask) + mask*trigger
            outputs = model(x)
            _, preds = torch.max(outputs, 1)
            count_success += torch.sum(c_t==preds.to('cpu')).item()
            count_all += len(x)
    return count_success/float(count_all)

def run_whitebox_attack(attack, data_loader, targeted, device, n_classes=4):
    x_adv_all, y_all = [], []
    for data in data_loader:
        x, y = data[0].to(device), data[1].to(device)
        if targeted:
            y = (y + torch.randint(low=1, high=n_classes, size=(len(y),), device=device))%n_classes
        x_adv = attack.execute(x, y, targeted=targeted)
        x_adv_all.append(x_adv)
        y_all.append(y)
    return torch.cat(x_adv_all), torch.cat(y_all)

def compute_attack_success(model, x_adv, y, batch_size, targeted, device):
    count_success = 0
    x_adv.to(device)
    y.to(device)
    with torch.no_grad():
        for i in range(0, len(x_adv), batch_size):
            x_batch = x_adv[i:i+batch_size]
            y_batch = y[i:i+batch_size]
            outputs = model(x_batch)
            _, preds = torch.max(outputs, 1)
            if not targeted:
                count_success += torch.sum(y_batch!=preds).detach()
            else:
                count_success += torch.sum(y_batch==preds).detach()
    return count_success/float(len(x_adv))

def save_as_im(x, outpath):
    """
    Used to store a numpy array (with values in [0,1] as an image).
    Outpath should specify the extension (e.g., end with ".jpg").
    """
    im = Image.fromarray((x*255.).astype(np.uint8)).convert('RGB')
    im.save(outpath)



def run_blackbox_attack(attack, data_loader, targeted, device, n_classes=4):
    """
    Runs the black-box attack on the labeled data returned by
    data_loader. If targeted==True, runs targeted attacks, where
    targets are selected at random (t=(c_x+randint(1, n_classes))%n_classes).
    Otherwise, runs untargeted attacks. 
    The function returns:
    1- Adversarially perturbed sampels (one per input sample).
    2- True labels in case of untargeted attacks, and target labels in
       case of targeted attacks.
    3- The number of queries made to create each adversarial example.
    """
    X , Y, Q = [] , [], []
    for data in data_loader:
        x, y = data[0].to(device), data[1].to(device)
        if targeted == True:
            y=(y+torch.randint_like(y,1, n_classes))%n_classes
        x_tags, queries = attack.execute(x,y,targeted)
        X.append(x_tags)
        Y.append(y)
        Q.append(queries)
    
    X = torch.cat(X)
    Y = torch.cat(Y)
    Q = torch.cat(Q)

    return X,Y,Q




################################################
#VISUALIZATION
###############################################


def visualize(file_path):
    per_layer = True if "sparse" in file_path else False
    # Storage
    layers = []
    accs = []
    pgd_srs = []
    nes_values = [[] for _ in range(4)]  # 4 NES configs
    
    with open(file_path, "r") as f:
        lines = [l.strip() for l in f if l.strip()]  # clean + skip empty lines

    # Process in chunks of 2 lines (first line = acc/pgd, second line = NES values)
    for i in range(0, len(lines), 2):
       
        if i == 0 and per_layer:
            continue

        line1 = lines[i]
        line2 = lines[i+1] if i+1 < len(lines) else ""

        # Match layer + acc + pgd_sr
        #match = re.match(r"(\w+)\s*\|\s*acc=([\d.]+)\s*\|\s*pgd_sr=([\d.]+)", line1)
        
        pattern = r"([\w.\-]+)\s*\|\s*acc=([\d.]+)\s*\|\s*pgd_sr=([\d.]+)"
        #pattern = r"([\w\-]+)\s*\|\s*acc=([\d.]+)\s*\|\s*pgd_sr=([\d.]+)"
        match = re.match(pattern, line1)
        
        if not match:
            continue

        layer, acc, pgd = match.groups()
        
        ####
        if per_layer == False:
            tmp = layer.split("_")[0]
            layer = tmp
        #exit(1)
        
        ####
        
        
        layer = layer.replace("_sparse", "")  # strip suffix
        if per_layer:
            layer = layer.split("_")[-1]
       
        layers.append(layer)
        accs.append(float(acc))
        pgd_srs.append(float(pgd))

        # Extract NES values from second line
        nes_matches = re.findall(r"nes_sr=([\d.]+),queries=\d+", line2)
        for j, val in enumerate(nes_matches):
            nes_values[j].append(float(val))


    if per_layer:
        plt.figure(figsize=(8, 5))
        #print(layers)
        #print(pgd_srs)
        #exit(1)
        plt.plot(layers, accs, marker="o", label="Bengin Accuracy")
        plt.plot(layers, pgd_srs, marker="o", label="PGD Success Rate")
   
        #plt.grid(True)
        #plt.tight_layout()
        #plt.savefig("layer_acc_pgd.png", dpi=300)
        #plt.close()
    
        # -------- Plot 2: NES curves --------
        nes_labels = [
            "momentum=0",
            #"NES (momentum=0, targeted=True)",
            "momentum=0.9",
            #"NES (momentum=0.9, targeted=True)"
        ]
    
        nes_values = [nes_values[0], nes_values[2]]
        #nes_labels=nes_labels[0,2]
    
        for j, nes_curve in enumerate(nes_values):
            plt.plot(layers, nes_curve, marker="o", label=nes_labels[j])
        plt.xlabel("Regularization Strengh")
        tag = file_path.split("_")[-2]

        plt.title("Adversarial Attack Success Rates and Accuracy - L1 Regularization - FC3 Layer")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"complete_graph_{tag}.png", dpi=300)
        plt.close()

    else:

        # -------- Plot 1: Accuracy vs PGD_SR --------
        plt.figure(figsize=(8, 5))
        #print(layers)
        #print(pgd_srs)
        #exit(1)
        plt.plot(layers, accs, marker="o", label="Bengin Accuracy")
        plt.plot(layers, pgd_srs, marker="o", label="PGD Success Rate")
        plt.xlabel("Layer")
        plt.title("PGD-Success Rate and Accuracy - Dropout Regularization")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("layer_acc_pgd.png", dpi=300)
        plt.close()

        # -------- Plot 2: NES curves --------
        nes_labels = [
            "momentum=0",
            #"NES (momentum=0, targeted=True)",
            "momentum=0.9",
            #"NES (momentum=0.9, targeted=True)"
        ]

        plt.figure(figsize=(8, 5))
        nes_values = [nes_values[0], nes_values[2]]
        #nes_labels=nes_labels[0,2]

        for j, nes_curve in enumerate(nes_values):
            plt.plot(layers, nes_curve, marker="o", label=nes_labels[j])
        plt.xlabel("Layer")
        plt.title("NES-Success Rate - Dropout Regularization")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("layer_nes.png", dpi=300)
        plt.close()




def parse_file(file_path):
    variants = []
    queries_m0 = []
    queries_m1 = []

    with open(file_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    # process in pairs of lines
    for i in range(0, len(lines), 2):
        variant_line = lines[i]
        results_line = lines[i+1]

        # variant name
        variant = variant_line.split("|")[0].strip()
        variant = variant.replace("_sparse", "")

        #if file_path.endswith("eval_results.txt"):  # ensure only first parse sets names
        variants.append(variant)

        # extract queries
        q_vals = re.findall(r"queries=(\d+)", results_line)
        q_vals = list(map(int, q_vals))

        if len(q_vals) >= 3:
            queries_m0.append(q_vals[0])   # 1st
            queries_m1.append(q_vals[2])   # 3rd
        else:
            queries_m0.append(None)
            queries_m1.append(None)

    return variants, queries_m0, queries_m1


def visualize_queries_dual(file1, file2, label1="L1", label2="Dropout"):
    variants1, m0_f1, m1_f1 = parse_file(file1)
    variants2, m0_f2, m1_f2 = parse_file(file2)

    assert variants1 == variants2, "Variants mismatch between files!"
    variants = variants1

    x = np.arange(len(variants))
    width = 0.2  # bar width

    plt.figure(figsize=(10,6))

    # File 1 bars

    #exit(1)
    plt.bar(x - width*1.5, m0_f1, width, label=f"{label1} (Momentum=0)")
    plt.bar(x - width*0.5, m1_f1, width, label=f"{label1} (Momentum=0.9)")

    # File 2 bars
    plt.bar(x + width*0.5, m0_f2, width, label=f"{label2} (Momentum=0)")
    plt.bar(x + width*1.5, m1_f2, width, label=f"{label2} (Momentum=0.9)")

    plt.xticks(x, variants, rotation=30, ha="right")
    plt.ylabel("Queries")
    plt.title("NES - Queries per Variant (Comparison)")
    plt.legend(ncol=2, loc="upper center")
    plt.grid(True, linestyle="--", alpha=0.6, axis="y")

    ylim = plt.ylim()
    plt.ylim(ylim[0], ylim[1] * 1.35)
    plt.tight_layout()

    plt.savefig("queries_vis_dual.png", dpi=300)
    plt.show()


'''
def visualize_queries():
    file_path = "eval_results.txt"   # replace with your file

    variants = []
    queries_m0 = []   # momentum = 0  (1st queries entry)
    queries_m1 = []   # momentum > 0 (3rd queries entry)

    with open(file_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    # process in pairs of lines
    for i in range(0, len(lines), 2):
        variant_line = lines[i]
        results_line = lines[i+1]

        # variant name = first token before '|'
        variant = variant_line.split("|")[0].strip()
        variant= variant.replace("_sparse", "")
        variants.append(variant)

        # extract all queries numbers
        q_vals = re.findall(r"queries=(\d+)", results_line)
        q_vals = list(map(int, q_vals))

        if len(q_vals) >= 3:
            queries_m0.append(q_vals[0])   # 1st
            queries_m1.append(q_vals[2])   # 3rd
        else:
            queries_m0.append(None)
            queries_m1.append(None)

    # ---- Plotting ----
    x = range(len(variants))
    x = np.arange(len(variants))
    width = 0.35  # bar width
    plt.figure(figsize=(8,5))
    
    plt.bar(x - width/2, queries_m0, width, label="Momentum = 0")
    plt.bar(x + width/2, queries_m1, width, label="Momentum = 0.9")

    plt.xticks(x, variants, rotation=30, ha="right")
    plt.ylabel("Queries")
    plt.title("NES - Queries per variant")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6, axis="y")

    ylim = plt.ylim()
    plt.ylim(ylim[0], ylim[1] * 1.35)
    plt.tight_layout()

    plt.savefig("queries_vis.png", dpi=300)
'''