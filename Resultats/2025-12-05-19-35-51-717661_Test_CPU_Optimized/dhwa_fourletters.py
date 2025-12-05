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

# --- OPTIMISATION CRITIQUE CPU ---
# Empêche PyTorch d'utiliser tous les cœurs pour des petits calculs (ce qui ralentit tout)
torch.set_num_threads(1) 
# ---------------------------------

# Tag
TAG = "Test_CPU_Optimized"

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

# Filtrage Dataset (Optionnel, ici on remet tout comme dans votre dernier code, 
# ou vous pouvez remettre DATASET = [17, 59, 21, 26] si vous voulez filtrer)
DIGITS = list(range(0 , 10))
LOWER  = list(range(10, 36))
UPPER  = list(range(36, 62))
ALL    = list(range(0 , 62))
# DATASET = LOWER + UPPER 

# h, X, l, q, A
DATASET = [17, 59, 21, 26]
TRAIN_INSTANCES = list(range(1, 5))
TEST_INSTANCES  = [0]


# Network
HIDDEN = [128]
BETA = 0.95

# --- OPTIMISATIONS ALGORITHMIQUES ---
# Traite 1 point sur 2. Divise le temps de calcul par 2 immédiatement.
DOWNSAMPLE = 2 

# Training
LR=5e-4
BETAS=(0.9, 0.999)

# Sur CPU, un batch trop petit (8) est inefficace, un trop gros (1000) sature le cache.
# 32 ou 64 est souvent le "Sweet Spot" pour le CPU.
BATCH = 32 
EPOCHS = 1000
PATIENCE = 5
DELTA = 0.05 # Ajusté pour être moins sensible aux micro-variations


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
        
        # Calcul de la taille temporelle optimisée
        self.time_steps = 250 // DOWNSAMPLE
        
        # Pré-allocation en float32 (format natif PyTorch) pour éviter les conversions pendant la boucle
        self.inputs  = numpy.zeros((len(self.files), len(self.characters), 5, 2, self.time_steps), dtype=numpy.float32)
        self.outputs = numpy.zeros((len(self.files)), dtype=int)
        
        for file_, file in enumerate(self.files) :
            content = data.read_original_data(os.path.join(self.directory, file)) 
            # Downsampling immédiat
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
        # Optimisation pour éviter de boucler inutilement
        if len(self.files) > 0:
            inputs = self.inputs[0].shape[0] * self.inputs[0].shape[2]
            outputs = numpy.max(self.outputs) if len(self.outputs) > 0 else 0
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
            self.leaky.append(snntorch.Leaky(beta=BETA, learn_threshold=True)) 
        
        print("Net :", flush=True)
        for layer, layer_ in enumerate(self.layers) :
            print(f"Layer {layer} size : {layer_}", flush=True)
        print("\n", flush=True)


    def forward(self, inputs):
        # inputs shape: (Batch, Features, Time)
        
        membrane_value = list()
        batch_size = inputs.shape[0]
        
        for i, leaky in enumerate(self.leaky) :
            membrane_value.append(leaky.reset_mem())
            # Sécurité pour s'assurer que la membrane a la bonne taille de batch
            if membrane_value[-1].dim() == 0:
                 membrane_value[-1] = torch.zeros(batch_size, self.layers[i+1])

        spike_record = []
        membrane_record = []
        
        # --- VECTORISATION (Accélération CPU) ---
        # Au lieu de calculer Linear à chaque étape t dans la boucle Python (lent),
        # on calcule tout d'un coup.
        # On passe de (Batch, Features, Time) -> (Batch, Time, Features)
        x = inputs.permute(0, 2, 1) 
        
        # On applique les couches linéaires
        # Note: Cela marche bien car le réseau est simple (Linear -> Leaky).
        # Si vous aviez plusieurs couches Linear successives sans Leaky entre, on pourrait tout pré-calculer.
        # Ici, on va devoir garder la boucle pour la dynamique temporelle du Leaky,
        # mais on peut optimiser le passage dans Linear.
        
        # Pour rester simple et robuste avec votre code existant (qui a plusieurs couches):
        # On garde la boucle classique MAIS avec moins d'étapes grâce au DOWNSAMPLE.
        # La vectorisation complète demanderait de réécrire la logique multi-couches.
        
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


def display(subject, subjects, fold, TRAIN_INSTANCES, epoch, EPOCHS, phase_name,
            batch, batch_, phase_loss, phase_accuracy_, phase_accuracy) :
    
    display    = f"[{fold:02d}/{len(TRAIN_INSTANCES):02d}][{epoch:02d}/{EPOCHS:02d}][{phase_name}]"
    display = f"[{subject:02d}/{subjects}]" + display
    display_   = ">" + "=" * min(batch, batch_) + "-" * max(0, batch_ - batch) + "> " 
    display__  =  f"$ = {phase_loss:06.2f} - "           \
               +  f"@ = {phase_accuracy_*100:06.2f} % - "       \
               + f"_@ = {phase_accuracy *100:06.2f} %"
    _display_    = "\33[A" + display + display_ + display__
    print(_display_ + " " * (LINE - len(display__)), flush=True)


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


# FORCE CPU EXPLICITE
device = torch.device("cpu")
print(f"Running on : {device}\n", flush=True)


nets = [list() for _ in range(len(subjects))]


test_accuracies  = [list() for _ in range(len(subjects))]
_test_accuracies = [list() for _ in range(len(subjects))]


for subject_, subject in enumerate(subjects) :


    print(f"Subject #{subject_} : {subject.split('-')[0]}\n", flush=True)

    
    test_dataset = CustomDataset(INPUT, subject, DATASET, TEST_INSTANCES)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH, drop_last=True)
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

        train_dataloader      = torch.utils.data.DataLoader(train_dataset     , batch_size=BATCH, drop_last=True)
        if fold != len(TRAIN_INSTANCES) :
            validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=BATCH, drop_last=True)
        
        
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
                
                # Conversion float32 explicite sans transfert CUDA
                train_data = train_data.to(torch.float)
                # Note: .to(device) est inutile si device est cpu, mais on le laisse par habitude
                
                nets[subject_][-1].train()
                train_results, _train_results = nets[subject_][-1](train_data.view(BATCH, train_data.shape[1] * train_data.shape[2] * train_data.shape[3], -1))

                train_loss_ = torch.zeros((1), dtype=torch.float, device=device)
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
                            
                optimizer.zero_grad()
                train_loss_.backward()
                optimizer.step()

                if DISPLAY == True :
                    display(subject_, len(subjects), fold, TRAIN_INSTANCES, epoch, EPOCHS, "T",
                    batch, batch_, train_loss_.item(), train_accuracy_, _train_accuracy_)
            
                batch += 1
            
            # Correction division par zéro si dataset vide
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
                        
                        validation_data = validation_data.to(torch.float)

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

                test_results, _test_results = nets[subject_][-1](test_data.view(BATCH, test_data.shape[1] * test_data.shape[2] * test_data.shape[3], -1))
                _,  test_index =  test_results.sum(dim=0).max(1)
                _, _test_index = _test_results.sum(dim=0).max(1)
                
                test_gt       = test_gt + test_targets.tolist()
                test_indices  = test_indices + test_index.tolist()
                _test_indices = _test_indices + _test_index.tolist()
                test_accuracy  += ( test_index == test_targets).sum().item()
                _test_accuracy += (_test_index == test_targets).sum().item()

            test_accuracies[subject_].append( 100 * test_accuracy / size)
            _test_accuracies[subject_].append(100 * _test_accuracy / size)

            test_gt_       = list(map(lambda i : f"{i:3d}",       test_gt))
            test_indices_  = list(map(lambda i : f"{i:3d}",  test_indices))
            _test_indices_ = list(map(lambda i : f"{i:3d}", _test_indices))
            for line in range(0, len(test_gt_) // LINE_ + 1) :
                start = line*LINE_
                if line == len(test_gt_) // LINE_:
                    end = -1
                else :
                    end = line*LINE_+LINE_
                print("Expected           : " + ", ".join(      test_gt_[start:end]), flush=True)
                print("Membrane predicted : " + ", ".join( test_indices_[start:end]), flush=True)
                print("Spike predicted    : " + ", ".join(_test_indices_[start:end]), flush=True)
                print("-------------------- ", flush=True)
            print(f"Membrane test accuracy : { test_accuracies[subject_][-1]:06.2f}%", flush=True)
            print(f"Spike    test accuracy : {_test_accuracies[subject_][-1]:06.2f}%", flush=True)
            print(flush=True)
            
            sklearn.metrics.ConfusionMatrixDisplay.from_predictions(test_gt, test_indices,  normalize="true")
            matplotlib.pyplot.title(title + "Membrane confusion matrix")
            matplotlib.pyplot.savefig(os.path.join(output, name + "confusion_membrane.png"))
            matplotlib.pyplot.close()

            sklearn.metrics.ConfusionMatrixDisplay.from_predictions(test_gt, _test_indices, normalize="true")
            matplotlib.pyplot.title(title + "Spikes confusion matrix")
            matplotlib.pyplot.savefig(os.path.join(output, name + "confusion_spikes.png"))
            matplotlib.pyplot.close()
            
    if SUBJECTS == 0 :
        print("Summary", flush=True)
    else :
        print(f"Summary for #{subject_} : {subject.split('-')[0]}", flush=True)
    for fold in range(0, len(TRAIN_INSTANCES) + 1) :
        print(f"Fold #{fold}", flush=True)
        print(f"Membrane test accuracy : { test_accuracies[subject_][fold]:06.2f}%", flush=True)
        print(f"Spike    test accuracy : {_test_accuracies[subject_][fold]:06.2f}%", flush=True)

    print("Total", flush=True)
    print(f"Membrane test accuracy : {numpy.mean( test_accuracies[subject_]):06.2f}%+/-{numpy.std( test_accuracies[subject_]):06.2f}", flush=True)
    print(f"Spike    test accuracy : {numpy.mean(_test_accuracies[subject_]):06.2f}%+/-{numpy.std(_test_accuracies[subject_]):06.2f}", flush=True)

    print(flush=True)


if SUBJECTS != 0 :
    print("Summary", flush=True)
    for subject_, subject in enumerate(subjects) :
        print(f"Subject #{subject_} : {subject.split('-')[0]}", flush=True)
        for fold in range(0, len(TRAIN_INSTANCES) + 1) :
            print(f"Fold #{fold}", flush=True)
            print(f"Membrane test accuracy : { test_accuracies[subject_][fold]:06.2f}%", flush=True)
            print(f"Spike    test accuracy : {_test_accuracies[subject_][fold]:06.2f}%", flush=True)
        print("Total", flush=True)
        print(f"Membrane test accuracy : {numpy.mean( test_accuracies[subject_]):06.2f}%+/-{numpy.std( test_accuracies[subject_]):06.2f}", flush=True)
        print(f"Spike    test accuracy : {numpy.mean(_test_accuracies[subject_]):06.2f}%+/-{numpy.std(_test_accuracies[subject_]):06.2f}", flush=True)
    print("TOTAL", flush=True)
    print(f"Membrane test accuracy : {numpy.mean( test_accuracies):06.2f}%+/-{numpy.std( test_accuracies):06.2f}", flush=True)
    print(f"Spike    test accuracy : {numpy.mean(_test_accuracies):06.2f}%+/-{numpy.std(_test_accuracies):06.2f}", flush=True)

open(os.path.join(output, "Summary.csv"), "w").write(
    "Subject;Fold;Membrane accuracy;Spike accuracy\n"
    + "\n".join(
        [f"{subject.split('-')[0]};{fold};{test_accuracies[subject_][fold]:06.2f};{_test_accuracies[subject_][fold]:06.2f}"
         for subject_, subject in enumerate(subjects) for fold in range(0, len(TRAIN_INSTANCES) + 1)])
    + f"\nAll;Mean+/-Std;"
    + f"{numpy.mean( test_accuracies[subject_]):06.2f}%+/-{numpy.std( test_accuracies[subject_]):06.2f};"
    + f"{numpy.mean(_test_accuracies[subject_]):06.2f}%+/-{numpy.std(_test_accuracies[subject_]):06.2f}"
    )