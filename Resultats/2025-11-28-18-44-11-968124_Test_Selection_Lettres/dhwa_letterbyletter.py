import os
import sys
import shutil
import datetime
import time  # Import pour le calcul du temps
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


# Tag
TAG = "Test_Selection_Lettres"

# Display
DISPLAY = False # On désactive l'affichage détaillé pour aller plus vite et ne pas polluer la console

# Paths

#### modifs ici ####
ROOT = "C:\\Users\\hatim\\OneDrive - Université de Bourgogne\\Bureau\\PFE\\pb_dossier\\DHWA"
#### fin modifs ici ####

DATA    = "Donnees"
RESULTS = "Resultats"
INPUT   = os.path.join(ROOT, DATA)
OUTPUT  = os.path.join(ROOT, RESULTS)

# Dataset
SUBJECTS = 1
# On teste sur le sujet 033 (difficile) ou 007 pour voir quelles lettres fonctionnent
SUBJECTS_ = ["033"] 

# On prépare la liste complète des 52 lettres
ALL_LETTERS = list(range(0, 62)) # 0-9 (chiffres) + 10-35 (minuscules) + 36-61 (majuscules)
# Pour l'expérience, on va se concentrer sur les LETTRES (minuscules et majuscules)
# On exclut les chiffres (0-9) pour l'instant
LETTERS_ONLY = list(range(10, 62)) 

TRAIN_INSTANCES = list(range(1, 5))
TEST_INSTANCES  = [0]

# Network (Optimisé)
HIDDEN = [128]
BETA = 0.95

# Training (Optimisé pour la vitesse)
LR = 1e-3
BETAS = (0.9, 0.999)
BATCH = 64
EPOCHS = 100 # On réduit encore un peu car apprendre 1 seule lettre est rapide
PATIENCE = 10
DELTA = 0.01
SUBSAMPLE = 3

# Misc
LINE  = 80


class Logger:
    def __init__(self, filename):
        self.console = sys.stdout
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
        
        time_steps = 250 // SUBSAMPLE + (1 if 250 % SUBSAMPLE != 0 else 0)
        self.inputs  = numpy.zeros((len(self.files), len(self.characters), 5, 2, time_steps))
        self.outputs = numpy.zeros((len(self.files)), dtype=int)
        
        for file_, file in enumerate(self.files) :
            content = data.read_original_data(os.path.join(self.directory, file)) 
            traj = content["trajectories"][self.characters, :, ::SUBSAMPLE, :2]
            self.inputs[file_] = numpy.swapaxes(numpy.nan_to_num(traj), 2, 3) 
            
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
        inputs = 0
        outputs = 0
        for file_, _ in enumerate(self.files) :
            inputs_ = self.inputs[file_].shape[0] * self.inputs[file_].shape[2]
            if inputs == 0 : inputs = inputs_
            outputs_ = self.outputs[file_]
            outputs = max(outputs, outputs_)
        return inputs, outputs + 1

class Net(torch.nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()
        self.layers = [inputs] + HIDDEN + [outputs]
        self.linear = torch.nn.ModuleList()
        self.leaky = torch.nn.ModuleList()
        for layer, _ in enumerate(self.layers[:-1]) :
            self.linear.append(torch.nn.Linear(self.layers[layer], self.layers[layer+1]))
            self.leaky.append(snntorch.Leaky(beta=BETA, learn_threshold=True)) 
    
    def forward(self, inputs):
        membrane_value = list()
        for leaky in self.leaky :
            membrane_value.append(leaky.reset_mem())
        spike_record = []
        membrane_record = []
        for step in range(inputs.shape[-1]) :
            numerical_value = inputs[:, :, step]
            for layer, _ in enumerate(self.layers[:-1]) :
                numerical_value = self.linear[layer](numerical_value)
                spike_value, membrane_value_ = self.leaky[layer](numerical_value, membrane_value[layer])
                membrane_value[layer] = membrane_value_
            spike_record.append(spike_value)
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


# --- DÉBUT DU SCRIPT PRINCIPAL ---

output = os.path.join(OUTPUT, "_".join([datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f"), TAG]))
if not os.path.exists(output):
    os.makedirs(output)

shutil.copy2(__file__, os.path.join(output, os.path.basename(__file__)))
sys.stdout = Logger(os.path.join(output, "Log.txt"))

files = [item for item in os.listdir(INPUT) if item.find("info") == -1]
if SUBJECTS == 0 :
    subjects = files
else :
    subjects = list()
    for file in files :
        if file.split("-")[0] in SUBJECTS_ :
            subjects.append(file)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Device used: {device}\n", flush=True)

# Initialisation du fichier CSV pour les résultats détaillés
csv_file = open(os.path.join(output, "Lettre_par_Lettre.csv"), "w")
csv_file.write("Subject;Letter_Index;Letter_Name;Execution_Time(s);Membrane_Acc_Mean;Membrane_Acc_Std;Spike_Acc_Mean;Spike_Acc_Std\n")

# Dictionnaire pour mapper les index aux noms de lettres
# 0-9: Chiffres, 10-35: a-z, 36-61: A-Z
def get_letter_name(index):
    if 0 <= index <= 9: return str(index)
    if 10 <= index <= 35: return chr(ord('a') + index - 10)
    if 36 <= index <= 61: return chr(ord('A') + index - 36)
    return "?"

# --- BOUCLE SUR LES SUJETS ---
for subject_, subject in enumerate(subjects) :

    print(f"Subject #{subject_} : {subject.split('-')[0]}", flush=True)
    
    # --- BOUCLE SUR CHAQUE LETTRE INDIVIDUELLEMENT ---
    # On va tester chaque lettre une par une pour voir laquelle est la plus discriminante
    for letter_idx in LETTERS_ONLY:
        
        start_time = time.time() # Démarrage du chrono pour cette lettre
        letter_name = get_letter_name(letter_idx)
        print(f"  > Testing Letter: {letter_name} (Index {letter_idx})...", end="", flush=True)
        
        # DATASET contient UNE SEULE lettre
        current_dataset = [letter_idx]
        
        # Listes pour stocker les résultats des 5 folds pour CETTE lettre
        fold_membrane_accs = []
        fold_spike_accs = []
        
        # --- BOUCLE SUR LES FOLDS (Validation Croisée) ---
        for fold in range(0, len(TRAIN_INSTANCES) + 1) :
            
            train_instances = list(TRAIN_INSTANCES)
            if fold != len(TRAIN_INSTANCES) :
                validation_instance = [TRAIN_INSTANCES[fold]]
                train_instances.remove(TRAIN_INSTANCES[fold])
            
            # Création des datasets pour cette lettre et ce fold
            train_dataset = CustomDataset(INPUT, subject, current_dataset, train_instances)
            if fold != len(TRAIN_INSTANCES) :
                validation_dataset = CustomDataset(INPUT, subject, current_dataset, validation_instance)
            else:
                # Pour le fold final, on utilise le dataset de test global (instance 0)
                # Mais attention, ici on teste juste la convergence, 
                # pour l'évaluation finale on utilise l'instance 0 comme Test set
                test_dataset = CustomDataset(INPUT, subject, current_dataset, TEST_INSTANCES)
                test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH, drop_last=False) # drop_last=False pour test

            train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH, drop_last=True)
            if fold != len(TRAIN_INSTANCES) :
                validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=BATCH, drop_last=True)
            
            # Création du réseau
            net = Net(*train_dataset.size()).to(device)
            optimizer = torch.optim.Adam(net.parameters(), lr=LR, betas=BETAS)
            loss_calculation = torch.nn.CrossEntropyLoss()
            early_stopping = EarlyStopping(patience=PATIENCE, delta=DELTA)
            
            # --- ENTRAÎNEMENT ---
            for epoch in range(1, EPOCHS + 1):
                net.train()
                train_loss = 0
                batch_count = 0
                
                for train_data, train_targets in iter(train_dataloader) :
                    train_data = train_data.to(torch.float).to(device)
                    train_targets = train_targets.to(device)
                    
                    train_results, _train_results = net(train_data.view(BATCH, train_data.shape[1] * train_data.shape[2] * train_data.shape[3], -1))
                    
                    loss = torch.zeros((1), dtype=torch.float, device=device)
                    for step in range(train_data.shape[-1]):
                        loss += loss_calculation(train_results[step], train_targets)
                    loss /= train_data.shape[-1]
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                    batch_count += 1
                
                # --- VALIDATION / EARLY STOPPING ---
                if fold != len(TRAIN_INSTANCES) :
                    net.eval()
                    val_loss = 0
                    val_batch_count = 0
                    with torch.no_grad():
                        for val_data, val_targets in iter(validation_dataloader):
                            val_data = val_data.to(torch.float).to(device)
                            val_targets = val_targets.to(device)
                            val_res, _ = net(val_data.view(BATCH, val_data.shape[1] * val_data.shape[2] * val_data.shape[3], -1))
                            loss = 0
                            for step in range(val_data.shape[-1]):
                                loss += loss_calculation(val_res[step], val_targets)
                            val_loss += (loss / val_data.shape[-1]).item()
                            val_batch_count += 1
                    
                    if val_batch_count > 0:
                        val_loss /= val_batch_count
                        early_stopping(val_loss, net)
                        if early_stopping.early_stop:
                            early_stopping.load_best_model(net)
                            break
                else:
                     # Pas de validation sur le dernier fold, on utilise le train loss
                     if batch_count > 0:
                        train_loss /= batch_count
                        early_stopping(train_loss, net)
                        if early_stopping.early_stop:
                             break

            # --- TEST SUR INSTANCE 0 (Une fois l'entraînement fini pour ce fold) ---
            # Note: Dans la logique originale, le test se fait à chaque fold.
            # Nous allons recréer le test_dataloader ici pour être sûr
            test_dataset = CustomDataset(INPUT, subject, current_dataset, TEST_INSTANCES)
            test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH, drop_last=False) # Important: False pour tout voir
            
            net.eval()
            correct_mem = 0
            correct_spike = 0
            total = 0
            with torch.no_grad():
                for data_, target_ in iter(test_dataloader):
                    data_ = data_.to(torch.float).to(device)
                    target_ = target_.to(device)
                    # Gestion de la taille du batch si dernier batch incomplet
                    curr_batch_size = target_.size(0)
                    
                    res_mem, res_spike = net(data_.view(curr_batch_size, data_.shape[1] * data_.shape[2] * data_.shape[3], -1))
                    
                    _, pred_mem = res_mem.sum(dim=0).max(1)
                    _, pred_spike = res_spike.sum(dim=0).max(1)
                    
                    correct_mem += (pred_mem == target_).sum().item()
                    correct_spike += (pred_spike == target_).sum().item()
                    total += curr_batch_size
            
            if total > 0:
                fold_membrane_accs.append(100.0 * correct_mem / total)
                fold_spike_accs.append(100.0 * correct_spike / total)
            
        # --- FIN DES FOLDS POUR CETTE LETTRE ---
        end_time = time.time()
        exec_time = end_time - start_time
        
        mean_mem = numpy.mean(fold_membrane_accs)
        std_mem = numpy.std(fold_membrane_accs)
        mean_spike = numpy.mean(fold_spike_accs)
        std_spike = numpy.std(fold_spike_accs)
        
        print(f" Done in {exec_time:.2f}s. Mem: {mean_mem:.2f}% (+/-{std_mem:.2f}) - Spike: {mean_spike:.2f}% (+/-{std_spike:.2f})", flush=True)
        
        # Écriture dans le CSV
        csv_file.write(f"{subject.split('-')[0]};{letter_idx};{letter_name};{exec_time:.4f};{mean_mem:.4f};{std_mem:.4f};{mean_spike:.4f};{std_spike:.4f}\n")
        csv_file.flush()

csv_file.close()
print("\n--- Fin de l'analyse lettre par lettre ---")