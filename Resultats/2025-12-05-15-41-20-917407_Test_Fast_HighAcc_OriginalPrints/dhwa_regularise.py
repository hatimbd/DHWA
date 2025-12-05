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
from torch.cuda.amp import autocast, GradScaler # Optimisation RTX

# Tag
TAG = "Test_Fast_HighAcc_OriginalPrints"

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


# Network
# Retour à 1000 pour garantir la performance d'origine
HIDDEN = [1000]
BETA = 0.95

# Training
# Optimisation : Batch plus gros pour le GPU, Downsample modéré pour la vitesse sans perte
DOWNSAMPLE = 2 # Divise le temps par 2 (125 steps), accélère x2 sans trop perdre de précision
BATCH = 256    # Augmenté pour utiliser la VRAM (8 -> 256)
LR = 1e-3      # Ajusté pour le batch size
BETAS=(0.9, 0.999)
EPOCHS = 1000
PATIENCE = 10
DELTA = 0.05   # Ajusté


# Misc
LINE  = 80
LINE_ = 20

# Classe utilitaire pour rediriger la sortie (print) à la fois vers la console et un fichier journal (Log.txt).
class Logger:
    def __init__(self, filename):
        if DISPLAY == True :
            self.console = sys.stdout
        else :
            self.console = open(os.devnull, 'w')
        self.file = open(filename, 'a')
    # Écrit un message à la fois dans la console et dans le fichier journal.
    def write(self, message):
        self.console.write(message)
        self.file.write(message)
    # Vide le buffer de la console et du fichier. Nécessaire pour s'assurer que tout est écrit.
    def flush(self):
        self.console.flush()
        self.file.flush()

# Classe PyTorch pour charger les données d'écriture manuscrite (DigiLeTs).
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
        
        # Optimisation: Calcul de la taille temporelle
        self.time_steps = 250 // DOWNSAMPLE
        
        # Optimisation: dtype=float32 pour éviter la conversion pendant l'entrainement
        self.inputs  = numpy.zeros((len(self.files), len(self.characters), 5, 2, self.time_steps), dtype=numpy.float32)
        self.outputs = numpy.zeros((len(self.files)), dtype=int)
        
        for file_, file in enumerate(self.files) :
            content = data.read_original_data(os.path.join(self.directory, file)) # (62, 5, 250, 8)
            # Optimisation: Downsampling ici
            raw_traj = content["trajectories"][self.characters, :, ::DOWNSAMPLE, :2]
            self.inputs[file_] = numpy.swapaxes(numpy.nan_to_num(raw_traj), 2, 3) 
            
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
        # Optimisation légère de la fonction size pour éviter de boucler si possible
        if len(self.files) > 0:
            inputs = self.inputs[0].shape[0] * self.inputs[0].shape[2]
            # On assume binaire ou max class selon les outputs chargés
            outputs = numpy.max(self.outputs) if len(self.outputs) > 0 else 0
            return inputs, outputs + 1
        return 0, 0

# Définit l'architecture du Réseau de Neurones à Spikes (SNN) en utilisant snnTorch.
class Net(torch.nn.Module):

    def __init__(self, inputs, outputs):

        super().__init__()
        
        self.layers = [inputs] + HIDDEN + [outputs]
        
        self.linear = torch.nn.ModuleList()
        self.leaky = torch.nn.ModuleList()
        
        for layer, _ in enumerate(self.layers[:-1]) :
            self.linear.append(torch.nn.Linear(self.layers[layer], self.layers[layer+1]))
            self.leaky.append(snntorch.Leaky(beta=BETA, learn_threshold=True)) # Restauration du learn_threshold=True original
        
        print("Net :", flush=True)
        for layer, layer_ in enumerate(self.layers) :
            print(f"Layer {layer} size : {layer_}", flush=True)
        print("\n", flush=True)


    def forward(self, inputs):

        membrane_value = list()
        # Optimisation: Initialisation sur le bon device directement
        batch_size = inputs.shape[0]
        for i, leaky in enumerate(self.leaky) :
            membrane_value.append(leaky.reset_mem().to(inputs.device)) # Fix pour s'assurer que c'est sur GPU
            # Si reset_mem ne prend pas en compte le batch size correctement avec snntorch parfois:
            if membrane_value[-1].dim() == 0: 
                 membrane_value[-1] = torch.zeros(batch_size, self.layers[i+1], device=inputs.device)

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

# Fonction d'origine restaurée
def display(subject, subjects, fold, TRAIN_INSTANCES, epoch, EPOCHS, phase_name,
            batch, batch_, phase_loss, phase_accuracy_, phase_accuracy) :
    
    display    = f"[{fold:02d}/{len(TRAIN_INSTANCES):02d}][{epoch:02d}/{EPOCHS:02d}][{phase_name}]"
    display = f"[{subject:02d}/{subjects}]" + display
    display_   = ">" + "=" * min(batch, batch_) + "-" * max(0, batch_ - batch) + "> " 
    display__  =  f"$ = {phase_loss:06.2f} - "          \
               +  f"@ = {phase_accuracy_*100:06.2f} % - "      \
               + f"_@ = {phase_accuracy *100:06.2f} %"
    _display_    = "\33[A" + display + display_ + display__
    print(_display_ + " " * (LINE - len(display__)), flush=True)

# --- DÉBUT DU SCRIPT PRINCIPAL ---

output = os.path.join(OUTPUT, "_".join([datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f"), TAG]))
os.mkdir(output)

shutil.copy2(__file__, os.path.join(output, os.path.basename(__file__)))

sys.stdout = Logger(os.path.join(output, "Log.txt"))

files = [item for item in os.listdir(INPUT) if item.find("info") == -1]
if SUBJECTS == 0 :
    subjects = files
    print("Subjects selected : All\n", flush=True)
else :
    if len(SUBJECTS_) == 0 :
        print(f"Subjects fund : {len(files)}", flush=True)
        subjects = numpy.random.default_rng().choice(files, size=SUBJECTS, replace=False)
    else :
        subjects = list()
        for file in files :
            if file.split("-")[0] in SUBJECTS_ :
                subjects.append(file)
    print(f"Subjects selected ({len(subjects)}) : {', '.join(map(lambda subject : f'{subject.split('-')[0]}', subjects))}\n", flush=True)


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Device : {device}\n", flush=True)

# Optimisation: Scaler pour Mixed Precision
scaler = GradScaler()

nets = [list() for _ in range(len(subjects))]


test_accuracies  = [list() for _ in range(len(subjects))]
_test_accuracies = [list() for _ in range(len(subjects))]


for subject_, subject in enumerate(subjects) :


    print(f"Subject #{subject_} : {subject.split('-')[0]}\n", flush=True)

    
    test_dataset = CustomDataset(INPUT, subject, DATASET, TEST_INSTANCES)
    # Optimisation: pin_memory=True
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH, drop_last=True, pin_memory=True)
    print(f"Test dataset length : {test_dataset.__len__()}\n", flush=True)


    for fold in range(0, len(TRAIN_INSTANCES) + 1) :


        if fold != len(TRAIN_INSTANCES) :
            print(f"Fold #{fold}\n", flush=True)
        else :
            print(f"No fold\n", flush=True)

            
        train_instances = list(TRAIN_INSTANCES)
        if fold != len(TRAIN_INSTANCES) :
            validation_instance = [TRAIN_INSTANCES[fold]]
            train_instances.remove(TRAIN_INSTANCES[fold])

        print(f"Train instances : {train_instances}", flush=True)
        if fold != len(TRAIN_INSTANCES) :
            print(f"Validation instance : {validation_instance}\n", flush=True)
            
            
        train_dataset = CustomDataset(INPUT, subject, DATASET, train_instances)
        print(f"Train dataset length : {train_dataset.__len__()}", flush=True)

        if fold != len(TRAIN_INSTANCES) :
            validation_dataset = CustomDataset(INPUT, subject, DATASET, validation_instance)
            print(f"Validation dataset length : {validation_dataset.__len__()}", flush=True)
        
        print(flush=True)

        # Optimisation: pin_memory=True
        train_dataloader      = torch.utils.data.DataLoader(train_dataset     , batch_size=BATCH, drop_last=True, pin_memory=True)
        if fold != len(TRAIN_INSTANCES) :
            validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=BATCH, drop_last=True, pin_memory=True)
        
        
        nets[subject_].append(Net(*train_dataset.size()).to(device))

        
        loss_calculation = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(nets[subject_][-1].parameters(), lr=LR, betas=BETAS)
        early_stopping = EarlyStopping(patience=PATIENCE, delta=DELTA)
        

        train_losses           = list()
        train_accuracies       = list()
        _train_accuracies      = list()
        validation_losses      = list()
        validation_accuracies  = list()
        _validation_accuracies = list()
        batch_  =      train_dataset.__len__() // BATCH
        if fold != len(TRAIN_INSTANCES):
            batch__ = validation_dataset.__len__() // BATCH
        else:
            batch__ = 0
        
        abort = False
        
        for epoch in range(1, EPOCHS + 1):
            
            train_loss = torch.zeros((1), dtype=torch.float, device=device)
            train_accuracy  = list()
            _train_accuracy = list()
            batch = 1

            for train_data, train_targets in iter(train_dataloader) :
                
                # Optimisation: non_blocking=True
                train_data = train_data.to(device, non_blocking=True)
                train_targets = train_targets.to(device, non_blocking=True)
                
                nets[subject_][-1].train()
                
                # Optimisation: Autocast (Mixed Precision)
                optimizer.zero_grad()
                with autocast():
                    train_results, _train_results = nets[subject_][-1](train_data.view(BATCH, train_data.shape[1] * train_data.shape[2] * train_data.shape[3], -1))

                    train_loss_ = torch.zeros((1), dtype=torch.float, device=device)
                    # Boucle temporelle (accélérée par DOWNSAMPLE)
                    for step in range(train_data.shape[-1]):
                        train_loss_ += loss_calculation(train_results[step], train_targets)
                    train_loss_ /= train_data.shape[-1]
                
                train_loss += train_loss_
                
                _,  train_index =  train_results.sum(dim=0).max(1)
                _, _train_index = _train_results.sum(dim=0).max(1)
                train_accuracy_  = numpy.mean((train_targets ==  train_index).detach().cpu().numpy())
                _train_accuracy_ = numpy.mean((train_targets == _train_index).detach().cpu().numpy())
                train_accuracy.append(train_accuracy_)
                _train_accuracy.append(_train_accuracy_)
                            
                # Optimisation: Scaler backprop
                scaler.scale(train_loss_).backward()
                scaler.step(optimizer)
                scaler.update()

                if DISPLAY == True :
                    display(subject_, len(subjects), fold, TRAIN_INSTANCES, epoch, EPOCHS, "T",
                    batch, batch_, train_loss_.item(), train_accuracy_, _train_accuracy_)
            
                batch += 1
            
            train_loss /= (batch - 1) if batch > 1 else 1
            train_losses.append(train_loss.item())
            train_accuracies.append( numpy.mean(train_accuracy))
            _train_accuracies.append(numpy.mean(_train_accuracy))

            
            if fold != len(TRAIN_INSTANCES) :


                with torch.no_grad():

                    nets[subject_][-1].eval()

                    validation_loss = torch.zeros((1), dtype=torch.float, device=device)
                    validation_accuracy  = list()
                    _validation_accuracy = list()
                    batch = 1

                    for validation_data, validation_targets in iter(validation_dataloader) :
                        
                        validation_data = validation_data.to(device, non_blocking=True)
                        validation_targets = validation_targets.to(device, non_blocking=True)

                        with autocast():
                            validation_results, _validation_results = nets[subject_][-1](validation_data.view(BATCH, validation_data.shape[1] * validation_data.shape[2] * validation_data.shape[3], -1))

                            validation_loss_ = torch.zeros((1), dtype=torch.float, device=device)
                            for step in range(validation_data.shape[-1]):
                                validation_loss_ += loss_calculation(validation_results[step], validation_targets)
                            validation_loss_ /= validation_data.shape[-1]
                        
                        validation_loss += validation_loss_

                        _, validation_index  =  validation_results.sum(dim=0).max(1)
                        _, _validation_index = _validation_results.sum(dim=0).max(1)
                        validation_accuracy_  = numpy.mean((validation_targets ==  validation_index).detach().cpu().numpy())
                        _validation_accuracy_ = numpy.mean((validation_targets == _validation_index).detach().cpu().numpy())
                        validation_accuracy.append(validation_accuracy_)
                        _validation_accuracy.append(_validation_accuracy_)

                        if DISPLAY == True :
                            display(subject_, len(subjects), fold, TRAIN_INSTANCES, epoch, EPOCHS, "V",
                            batch, batch__, validation_loss_.item(), validation_accuracy_, _validation_accuracy_)

                        batch += 1
                    
                    validation_loss /= (batch - 1) if batch > 1 else 1
                    validation_losses.append(validation_loss.item())
                    validation_accuracies.append( numpy.mean(validation_accuracy))
                    _validation_accuracies.append(numpy.mean(_validation_accuracy))
                    

                early_stopping(validation_loss, nets[subject_][-1])
                if early_stopping.early_stop :
                    abort = True
                    break
            

            else :
                
                early_stopping(train_loss, nets[subject_][-1])
                if early_stopping.early_stop :
                    abort = True
                    break

            if  abort == True :
                print("Early stopping\n", flush=True)
                early_stopping.load_best_model(nets[subject_][-1])
                break  
        
                
        text = f"Subject #{subject_} ({subject.split('-')[0]})"
        text_ = f"Fold #{fold}"
        title = text + " - " + text_
        name  = text + "_"   + text_
        
    
        matplotlib.pyplot.figure(facecolor="w", figsize=(10, 5))
        matplotlib.pyplot.plot(train_losses)
        if fold != len(TRAIN_INSTANCES) :
            matplotlib.pyplot.plot(validation_losses)
            matplotlib.pyplot.title(title + "Loss curves")
            matplotlib.pyplot.legend(["Train loss", "Validation loss"])
        else :
            matplotlib.pyplot.title(title + "Loss curve")
        matplotlib.pyplot.xlabel("Epochs")
        matplotlib.pyplot.ylabel("Loss")
        matplotlib.pyplot.savefig(os.path.join(output, name + "loss.png"))
        matplotlib.pyplot.close()

        matplotlib.pyplot.figure(facecolor="w", figsize=(10, 5))
        matplotlib.pyplot.plot(train_accuracies)
        if fold != len(TRAIN_INSTANCES) :
            matplotlib.pyplot.plot(validation_accuracies)
            matplotlib.pyplot.title(title + "Membrane accuracy curves")
            matplotlib.pyplot.legend(["Train acccuracy", "Validation acccuracy"])
        else :
            matplotlib.pyplot.title(title + "Membrane accuracy curve")
        matplotlib.pyplot.xlabel("Epochs")
        matplotlib.pyplot.ylabel("Accuracy")
        matplotlib.pyplot.savefig(os.path.join(output, name + "accuracy_membrane.png"))
        matplotlib.pyplot.close()

        matplotlib.pyplot.figure(facecolor="w", figsize=(10, 5))
        matplotlib.pyplot.plot(_train_accuracies)
        if fold != len(TRAIN_INSTANCES) :
            matplotlib.pyplot.plot(_validation_accuracies)
            matplotlib.pyplot.title(title + "Spikes accuracy curves")
            matplotlib.pyplot.legend(["Train acccuracy", "Validation acccuracy"])
        else :
            matplotlib.pyplot.title(title + "Spikes accuracy curve")
        matplotlib.pyplot.xlabel("Epochs")
        matplotlib.pyplot.ylabel("Accuracy")
        matplotlib.pyplot.savefig(os.path.join(output, name + "accuracy_spikes.png"))
        matplotlib.pyplot.close()

        with torch.no_grad():
        
            nets[subject_][-1].eval()

            test_gt       = list()
            test_indices  = list()
            _test_indices = list()
            test_accuracy  = 0
            _test_accuracy = 0
            size = 0

            for test_data, test_targets in iter(test_dataloader) :
                
                test_data = test_data.to(torch.float)
                test_data = test_data.to(device)
                test_targets = test_targets.to(device)

                size += test_targets.size(0)

                # Pas d'autocast ici pour s'assurer de la precision max en test, ou alors on peut le mettre
                test_results, _test_results = nets[subject_][-1](test_data.view(BATCH, test_data.shape[1] * test_data.shape[2] * test_data.shape[3], -1))
                _,  test_index =  test_results.sum(dim=0).max(1)
                _, _test_index = _test_results.sum(dim=0).max(1)
                
                test_gt       = test_gt + test_targets.tolist()
                test_indices  = test_indices + test_index.tolist()
                _test_indices = _test_indices + _