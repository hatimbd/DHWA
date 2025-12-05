import os
import sys
import shutil
import datetime
import re
import random
import socket
import matplotlib.pyplot
import numpy
import sklearn.metrics
import torch.utils.data
import torch.nn
import snntorch
import data
from torch.cuda.amp import autocast, GradScaler # Pour l'accélération matérielle RTX

# Tag
TAG = "Test_UltraFast_RTX4060"

# Display
DISPLAY = True

# Paths
ROOT = "C:\\Users\\hatim\\OneDrive - Université de Bourgogne\\Bureau\\PFE\\pb_dossier\\DHWA"

DATA    = "Donnees"
RESULTS = "Resultats"

INPUT  = os.path.join(ROOT, DATA   )
OUTPUT = os.path.join(ROOT, RESULTS)


# Dataset
SUBJECTS = 1 
SUBJECTS_ = ["033"] 

# Indices demandés : h(17), X(59), l(21), q(26)
DATASET = [17, 59, 21, 26] 

TRAIN_INSTANCES = list(range(1, 5)) 
TEST_INSTANCES  = [0]


# --- OPTIMISATIONS ---
# 1. Réduction temporelle drastique (50 étapes au lieu de 250 -> 5x plus rapide)
DOWNSAMPLE = 5 

# Network
HIDDEN = [256] 
BETA = 0.90 # Légèrement réduit pour une décroissance plus rapide (dynamique plus vive sur moins de pas)

# Training
LR = 2e-3 # Learning rate augmenté car le Batch est énorme
BETAS = (0.9, 0.999)

# 2. Utilisation massive de la VRAM (0.8GB -> ~4GB potentiels sur 8GB)
BATCH = 1024 

EPOCHS = 1000
PATIENCE = 15
DELTA = 0.001 

# Misc
LINE  = 80
LINE_ = 20

class Logger:
    def __init__(self, filename):
        if DISPLAY == True :
            self.console = sys.stdout
        else :
            self.console = open(os.devnull, 'w')
        self.file = open(filename, 'a')
    
    def write(self, message):
        self.console.write(message)
        self.file.write(message)
    
    def flush(self):
        self.console.flush()
        self.file.flush()

class CustomDataset(torch.utils.data.IterableDataset) :

    def __init__(self, directory, subject, characters, instances) :
        self.directory = directory
        self.characters = characters
        self.instances = instances
        self.files = [item for item in os.listdir(self.directory) if item.find("info") == -1]
        
        number = len(self.files) - 1
        if subject is not None :
            self.files = numpy.repeat(self.files, [number if file_ == subject else 1 for file_ in self.files], axis=0)
        
        random.shuffle(self.files)
        
        self.time_steps = 250 // DOWNSAMPLE
        
        # Pré-allocation mémoire
        # Utilisation de float32 directement pour éviter les conversions pendant l'entrainement
        self.inputs  = numpy.zeros((len(self.files), len(self.characters), 5, 2, self.time_steps), dtype=numpy.float32)
        self.outputs = numpy.zeros((len(self.files)), dtype=int)
        
        for file_, file in enumerate(self.files) :
            content = data.read_original_data(os.path.join(self.directory, file)) 
            
            # Downsampling et conversion immédiate
            raw_trajectories = content["trajectories"][self.characters, :, ::DOWNSAMPLE, :2] 
            self.inputs[file_] = numpy.swapaxes(numpy.nan_to_num(raw_trajectories), 2, 3) 
            
            if subject is None :
                self.outputs[file_] = int(content["wid"])
            else :
                self.outputs[file_] = int(content["wid"] == subject.split("-")[0])

    def __len__(self) :
        return len(self.files) * len(self.instances)

    def __iter__(self) :
        for file_, _ in enumerate(self.files) :
            for instance in self.instances :
                yield self.inputs[file_][:, instance:instance+1, :, :], self.outputs[file_] 
                
    def size(self) :
        # Calcul rapide sans itération si possible, sinon itération minimale
        if len(self.files) > 0:
            inputs = self.inputs[0].shape[0] * self.inputs[0].shape[2]
            # Pour les outputs, on suppose binaire (0 ou 1) vu le code précédent
            outputs = 1 
            return inputs, outputs + 1
        return 0, 0

class Net(torch.nn.Module):

    def __init__(self, inputs, outputs):
        super().__init__()
        self.layers = [inputs] + HIDDEN + [outputs]
        self.linear = torch.nn.ModuleList()
        self.leaky = torch.nn.ModuleList()
        
        for layer, _ in enumerate(self.layers[:-1]) :
            self.linear.append(torch.nn.Linear(self.layers[layer], self.layers[layer+1]))
            # learn_threshold=False accélère légèrement le graphe de calcul
            self.leaky.append(snntorch.Leaky(beta=BETA, learn_threshold=False)) 
        
        print("Net initialized.", flush=True)


    def forward(self, inputs):
        # Initialisation rapide des membranes
        membrane_value = []
        batch_size = inputs.size(0)
        
        # On initialise les membranes à 0 sur le bon device directement
        for i in range(len(self.layers) - 1):
             membrane_value.append(torch.zeros(batch_size, self.layers[i+1], device=inputs.device))

        spike_record = []
        membrane_record = []

        # Boucle temporelle (c'est le bottleneck principal des SNN)
        # inputs shape: [BATCH, Features, Time] -> on itère sur Time
        for step in range(inputs.shape[-1]) :
            numerical_value = inputs[:, :, step]
            
            for layer in range(len(self.linear)):
                numerical_value = self.linear[layer](numerical_value)
                spike_value, membrane_value_ = self.leaky[layer](numerical_value, membrane_value[layer])
                membrane_value[layer] = membrane_value_
                # numerical_value devient le spike pour la couche suivante (sauf si c'est la sortie, à voir selon archi SNNTorch)
                # Note: Dans votre code original, numerical_value n'était pas mis à jour par le spike, 
                # c'était un réseau dense -> dense -> leaky. Je garde la logique originale.
            
            spike_record.append(spike_value)
            # On n'a pas forcément besoin de tout l'historique de membrane pour la backprop si on utilise des surrogates simples,
            # mais on le garde pour l'affichage/cohérence.
            membrane_record.append(membrane_value_)

        return torch.stack(membrane_record, dim=0), torch.stack(spike_record, dim=0)

class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_model_state = None
        
    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict()
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = model.state_dict()
            self.counter = 0

    def load_best_model(self, model):
        model.load_state_dict(self.best_model_state)

def display(subject, subjects, fold, TRAIN_INSTANCES, epoch, EPOCHS, phase_name,
            batch, batch_, phase_loss, phase_accuracy_, phase_accuracy) :
    
    display    = f"[{fold:02d}/{len(TRAIN_INSTANCES):02d}][{epoch:02d}/{EPOCHS:02d}][{phase_name}]"
    display = f"[{subject:02d}/{subjects}]" + display
    # Simplification de l'affichage pour éviter le lag console
    print(f"\r{display} Loss:{phase_loss:06.4f} Acc_M:{phase_accuracy_*100:05.2f}% Acc_S:{phase_accuracy *100:05.2f}%", end="", flush=True)

# --- DÉBUT DU SCRIPT PRINCIPAL ---

output = os.path.join(OUTPUT, "_".join([datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f"), TAG]))
os.mkdir(output)
shutil.copy2(__file__, os.path.join(output, os.path.basename(__file__)))
sys.stdout = Logger(os.path.join(output, "Log.txt"))

files = [item for item in os.listdir(INPUT) if item.find("info") == -1]
if SUBJECTS == 0 :
    subjects = files
else :
    if len(SUBJECTS_) == 0 :
        subjects = numpy.random.default_rng().choice(files, size=SUBJECTS, replace=False)
    else :
        subjects = [f for f in files if f.split("-")[0] in SUBJECTS_]
    print(f"Subjects selected: {len(subjects)}\n", flush=True)

device = torch.device("cuda") # Force CUDA vu le screenshot
print(f"Using device: {device} ({torch.cuda.get_device_name(0)})", flush=True)

# 3. Initialisation du Scaler pour Mixed Precision (FP16)
scaler = GradScaler()

nets = [list() for _ in range(len(subjects))]
test_accuracies  = [list() for _ in range(len(subjects))]
_test_accuracies = [list() for _ in range(len(subjects))]

for subject_, subject in enumerate(subjects) :

    print(f"Processing Subject #{subject_} : {subject.split('-')[0]}\n", flush=True)
    
    # Dataset Test
    test_dataset = CustomDataset(INPUT, subject, DATASET, TEST_INSTANCES)
    # num_workers=0 sur Windows peut parfois être plus stable, mais 2 ou 4 aide si le CPU suit.
    # pin_memory=True est CRUCIAL pour le transfert RAM -> VRAM
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH, drop_last=False, pin_memory=True) 

    for fold in range(0, len(TRAIN_INSTANCES) + 1) :
        
        # Gestion des folds
        train_instances = list(TRAIN_INSTANCES)
        if fold != len(TRAIN_INSTANCES) :
            validation_instance = [TRAIN_INSTANCES[fold]]
            train_instances.remove(TRAIN_INSTANCES[fold])
            print(f"Fold #{fold} (Val: {validation_instance})", flush=True)
        else :
            print(f"Full Training (No Fold)", flush=True)

        # Datasets & Loaders
        train_dataset = CustomDataset(INPUT, subject, DATASET, train_instances)
        
        # DataLoader Optimisé
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=BATCH, 
            drop_last=True, 
            shuffle=False, # IterableDataset ne supporte pas shuffle=True standard facilement, le shuffle est fait dans __init__
            pin_memory=True
        )

        if fold != len(TRAIN_INSTANCES) :
            validation_dataset = CustomDataset(INPUT, subject, DATASET, validation_instance)
            validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=BATCH, drop_last=False, pin_memory=True)
        
        # Model Setup
        nets[subject_].append(Net(*train_dataset.size()).to(device))
        
        loss_calculation = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(nets[subject_][-1].parameters(), lr=LR, betas=BETAS)
        early_stopping = EarlyStopping(patience=PATIENCE, delta=DELTA)
        
        # Metrics storage
        train_losses = []
        
        abort = False
        
        for epoch in range(1, EPOCHS + 1):
            
            nets[subject_][-1].train()
            total_loss = 0
            correct_mem = 0
            correct_spk = 0
            total_samples = 0
            batch_count = 0

            for train_data, train_targets in iter(train_dataloader) :
                
                # Transfert GPU optimisé (non_blocking=True grâce à pin_memory)
                train_data = train_data.to(device, non_blocking=True) 
                train_targets = train_targets.to(device, non_blocking=True)
                
                optimizer.zero_grad(set_to_none=True) # set_to_none est plus rapide que zero_grad standard

                # --- MIXED PRECISION TRAINING ---
                with autocast():
                    # Flatten: (BATCH, Features, Time)
                    flattened_input = train_data.view(train_data.size(0), -1, train_data.size(-1))
                    
                    train_results, _train_results = nets[subject_][-1](flattened_input)

                    # Calcul de la Loss (moyenne temporelle)
                    loss = torch.zeros((1), device=device)
                    # Vectorisation possible ici ? Difficile avec CrossEntropyLoss standard sur seq time
                    # On boucle mais c'est sur GPU, c'est rapide
                    for step in range(train_data.shape[-1]):
                        loss += loss_calculation(train_results[step], train_targets)
                    loss /= train_data.shape[-1]
                
                # Backward Pass avec Scaler
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                # Metrics Calculation (Detach pour ne pas garder le graph)
                total_loss += loss.item()
                
                _,  train_index =  train_results.sum(dim=0).max(1)
                _, _train_index = _train_results.sum(dim=0).max(1)
                
                correct_mem += (train_targets == train_index).sum().item()
                correct_spk += (train_targets == _train_index).sum().item()
                total_samples += train_targets.size(0)
                batch_count += 1
            
            # Epoch Summary
            avg_loss = total_loss / batch_count if batch_count > 0 else 0
            acc_mem = correct_mem / total_samples if total_samples > 0 else 0
            acc_spk = correct_spk / total_samples if total_samples > 0 else 0
            train_losses.append(avg_loss)

            if DISPLAY:
                print(f"\r[{epoch:03d}/{EPOCHS}] Loss:{avg_loss:.4f} Acc:{acc_mem*100:.1f}%", end="")

            # Validation logic
            val_loss_val = 0
            if fold != len(TRAIN_INSTANCES) :
                with torch.no_grad():
                    nets[subject_][-1].eval()
                    v_loss = 0
                    v_count = 0
                    for val_d, val_t in validation_dataloader:
                        val_d = val_d.to(device, non_blocking=True)
                        val_t = val_t.to(device, non_blocking=True)
                        
                        with autocast():
                            val_res, _ = nets[subject_][-1](val_d.view(val_d.size(0), -1, val_d.size(-1)))
                            l = 0
                            for s in range(val_d.shape[-1]):
                                l += loss_calculation(val_res[s], val_t)
                            v_loss += (l / val_d.shape[-1]).item()
                        v_count += 1
                    
                    val_loss_val = v_loss / v_count if v_count > 0 else 0
                
                early_stopping(val_loss_val, nets[subject_][-1])
            else:
                early_stopping(avg_loss, nets[subject_][-1])

            if early_stopping.early_stop :
                print(f"\nEarly stopping at epoch {epoch}", flush=True)
                early_stopping.load_best_model(nets[subject_][-1])
                break
        
        print("\n", flush=True) # Nouvelle ligne après l'époque

        # Evaluation Finale du Fold
        with torch.no_grad():
            nets[subject_][-1].eval()
            correct_mem = 0
            correct_spk = 0
            total = 0
            
            all_preds = []
            all_targets = []

            for t_d, t_t in test_dataloader:
                t_d = t_d.to(device)
                t_t = t_t.to(device)
                
                with autocast():
                    res_m, res_s = nets[subject_][-1](t_d.view(t_d.size(0), -1, t_d.size(-1)))
                
                _, pred_m = res_m.sum(dim=0).max(1)
                _, pred_s = res_s.sum(dim=0).max(1)
                
                correct_mem += (pred_m == t_t).sum().item()
                correct_spk += (pred_s == t_t).sum().item()
                total += t_t.size(0)
                
                all_preds.extend(pred_m.cpu().numpy())
                all_targets.extend(t_t.cpu().numpy())

            acc_m = (correct_mem / total * 100) if total > 0 else 0
            acc_s = (correct_spk / total * 100) if total > 0 else 0
            
            test_accuracies[subject_].append(acc_m)
            _test_accuracies[subject_].append(acc_s)
            
            print(f"Final Test - Membrane: {acc_m:.2f}% | Spike: {acc_s:.2f}%", flush=True)
            
            # Matrice de confusion (seulement si données présentes)
            if total > 0:
                sklearn.metrics.ConfusionMatrixDisplay.from_predictions(all_targets, all_preds, normalize="true")
                name = f"{subject.split('-')[0]}_Fold{fold}_"
                matplotlib.pyplot.title(f"Confusion Matrix {name}")
                matplotlib.pyplot.savefig(os.path.join(output, name + "confusion.png"))
                matplotlib.pyplot.close()

# Sauvegarde CSV Final
print("Saving Summary...", flush=True)
with open(os.path.join(output, "Summary.csv"), "w") as f:
    f.write("Subject;Fold;Membrane accuracy;Spike accuracy\n")
    for s_idx, s in enumerate(subjects):
        name = s.split('-')[0]
        for fold in range(len(test_accuracies[s_idx])):
            f.write(f"{name};{fold};{test_accuracies[s_idx][fold]:.2f};{_test_accuracies[s_idx][fold]:.2f}\n")
    
    # Global stats
    all_m = [item for sublist in test_accuracies for item in sublist]
    all_s = [item for sublist in _test_accuracies for item in sublist]
    if all_m:
        f.write(f"All;Mean;{numpy.mean(all_m):.2f};{numpy.mean(all_s):.2f}\n")