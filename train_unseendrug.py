import os
import datetime
import argparse
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from collections import defaultdict
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from utils import *
from dataset_processed import Graph_Bert_Dataset_fine_tune
from model_motif_level import *
from kgnn import *
from mol_fingerprint_extractor import MolFingerprintMLPExtractor
from AsyMNfusion import AsyMNFusion
from evidential_tf import DirichletAlphasFromLogits, DirichletEvidentialLoss

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

TRAIN_SEEDS = [22]

parser = argparse.ArgumentParser()
parser.add_argument("--neighbor_sample_size", choices=[4, 5, 6, 10, 16], type=int, default=6)
parser.add_argument("--event_num", type=int, default=4)
parser.add_argument("--n_drug", type=int, default=2459)
parser.add_argument("--dropout", type=float, default=0.3)
parser.add_argument("--embedding_num", type=int, choices=[128, 64, 256, 32], default=256)
parser.add_argument("--redundancy_weight", type=float, default=0.005, help="Modal redundancy penalty weight")
parser.add_argument("--dirichlet_lam", type=float, default=0.1, help="The weight λ of the KL term in the Dirichlet evidence loss")
parser.add_argument("--data_setting", type=str, default="unseendrug-twodrugs", choices=["unseendrug-onedrug", "unseendrug-twodrugs"])
parser.add_argument("--learning_rate", type=float, default=9e-5)
parser.add_argument("--only_seed", type=int, default=None)
parser.add_argument("--save_dir_suffix", type=str, default="")
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--initial_epoch", type=int, default=0)
parser.add_argument("--early_stop_patience", type=int, default=3)
parser.add_argument('--stats_a', type=str, default='./data/stats_a_fullcov.npz')
parser.add_argument('--stats_b', type=str, default='./data/stats_b_fullcov.npz')
parser.add_argument('--mi_temperature', type=float, default=1.0)
parser.add_argument('--local_temperature', type=float, default=1.0)
parser.add_argument('--responsibility_temperature', type=float, default=0.2)
args = parser.parse_args()

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

def input_solver1(sample, sample1, sample2, sample3, spatial_pos1, spatial_pos2, sample4, sample5, sample6):
    return {
        'molecule_sequence1': sample, 'molecule_sequence2': sample1,
        'drugsa_index': sample2, 'drugsb_index': sample3,
        'spatial_pos1': spatial_pos1, 'spatial_pos2': spatial_pos2,
        'smiles1': sample4, 'smiles2': sample5,
    }, sample6

script_path = os.path.abspath(__file__)
script_directory = os.path.dirname(script_path)
dataFolder = './data'
MODEL_RUN_SUFFIX = args.data_setting

print(f"=== {args.data_setting} ===")
_tr_rel = "classification/unseendrug/tr_dataset-3.csv"
_val_one_rel = "classification/unseendrug/val_dataset_unseen_onedrug.csv"
_val_two_rel = "classification/unseendrug/val_dataset_unseen_twodrugs.csv"

tr_df = pd.read_csv(os.path.join(dataFolder, _tr_rel))
val_one_df = pd.read_csv(os.path.join(dataFolder, _val_one_rel))
val_two_df = pd.read_csv(os.path.join(dataFolder, _val_two_rel))

tokenizer = Mol_Tokenizer(dataFolder + '/token_id.json')
map_dict = np.load(dataFolder + '/preprocessed_drug_info.npy', allow_pickle=True).item()

dict111 = {}
train_df = pd.read_csv(dataFolder + '/databases.csv')
for item in train_df['Drug']:
    dict111[item] = dict111.get(item, len(dict111))

train_ds, unseen_one_ds, unseen_two_ds = Graph_Bert_Dataset_fine_tune(
    tr_df, val_one_df, val_two_df,
    dict111, tokenizer=tokenizer, map_dict=map_dict, batch_size=64
).get_data()

train_dataset = train_ds.map(input_solver1)
unseen_one_dataset = unseen_one_ds.map(input_solver1)
unseen_two_dataset = unseen_two_ds.map(input_solver1)

if args.data_setting == "unseendrug-onedrug":
    val_dataset = unseen_one_dataset
    test_dataset = unseen_one_dataset
    print(f"  train: {_tr_rel}\n  val/test: {_val_one_rel}")
else:
    val_dataset = unseen_two_dataset
    test_dataset = unseen_two_dataset
    print(f"  train: {_tr_rel}\n  val/test: {_val_two_rel}")

#parameter
num_layers = 4
num_heads = 8
d_model = 128 * 2
dff = d_model
input_vocab_size = tokenizer.get_vocab_size
dropout_rate = 0.3

#KG construction
def read_dataset(drug_name_id, num):
    kg, tails, relations, drug_name_id1 = defaultdict(list), {}, {}, {}
    filename = dataFolder + "/zheng_dataset" + str(num) + ".txt"
    with open(filename, encoding="utf8") as reader:
        for line in reader:
            string = line.rstrip().split()
            head = string[0]
            tail = string[2]
            relation = string[1]
            drug_name_id1[head] = drug_name_id[head]
            if num == 3: tails[tail] = drug_name_id[tail]
            else: tails[tail] = tails.get(tail, len(tails))
            relations[relation] = relations.get(relation, len(relations))
            kg[drug_name_id1[head]].append((tails[tail], relations[relation]))
    return kg, len(tails), len(relations), drug_name_id1

drug_name = [dict111[name] for name in dict111]
tr_dataset1_kg, tr_dataset1_tail_len, tr_dataset1_relation_len, dict1 = read_dataset(dict111, 1)
dataset2_kg, dataset2_tail_len, dataset2_relation_len, dict2 = read_dataset(dict111, 2)

dataset = {"dataset1": tr_dataset1_kg, "dataset2": dataset2_kg}
tail_len = {"dataset1": tr_dataset1_tail_len, "dataset2": dataset2_tail_len}
relation_len = {"dataset1": tr_dataset1_relation_len, "dataset2": dataset2_relation_len}

class SaveModelWithMetrics(Callback):
    def __init__(self, base_path, monitor='val_loss', verbose=2, save_best_only=True, mode='min'):
        super(SaveModelWithMetrics, self).__init__()
        self.base_path = base_path
        self.monitor = monitor
        self.verbose = verbose
        self.save_best_only = save_best_only
        self.mode = mode
        self.best_value = None
        if not os.path.exists(self.base_path): os.makedirs(self.base_path)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current_value = logs.get(self.monitor)
        current_acc = logs.get('val_sparse_categorical_accuracy')
        redundancy_loss = logs.get('modal_redundancy_loss', None)

        if redundancy_loss is not None:
            print(f"  [Redundancy] modal_redundancy_loss: {redundancy_loss:.6f}")

        if current_value is None: return

        if self.best_value is None:
            self.best_value = current_value
            self._save_model(epoch, current_value, current_acc)
        else:
            improved = (self.mode == 'min' and current_value < self.best_value) or \
                       (self.mode == 'max' and current_value > self.best_value)
            if improved:
                self.best_value = current_value
                self._save_model(epoch, current_value, current_acc)

    def _save_model(self, epoch, value, acc):
        acc_str = "nan" if acc is None else "{:.4f}".format(acc)
        value_str = "{:.4f}".format(value)
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"mymodel_epoch{epoch+1}_{self.monitor}{value_str}_acc{acc_str}_time{timestamp}.tf"
        filepath = os.path.join(self.base_path, filename)
        self.model.save(filepath, overwrite=True, include_optimizer=True)


def train_one_seed(seed):
    tf.keras.utils.set_random_seed(seed)
    tf.keras.backend.clear_session()

    #KG
    input1_layer = Input(shape=(None,), name="drugsa_index")
    input2_layer = Input(shape=(None,), name="drugsb_index")
    gnn1_output = GNN1(dataset, tail_len, relation_len, args, dict1, drug_name)(input1_layer, input2_layer)
    gnn2_output = GNN2(dataset, tail_len, relation_len, args, dict2, drug_name)(gnn1_output)
    fusion_layer = FusionLayer(args)
    kg_a, kg_b, kg_pair = fusion_layer(gnn2_output)

    #Motif
    motif_inputs = Input(shape=(None,), name="molecule_sequence")
    motif_spatial_pos = Input(shape=(None, None), dtype=tf.int32, name="spatial_pos")
    Outseq_motif, *_, encoder_padding_mask_motif = EncoderModel_motif(
        num_layers=num_layers, d_model=d_model, dff=dff, num_heads=num_heads,
        input_vocab_size=input_vocab_size, max_graph_distance=8,
    )([motif_inputs, motif_spatial_pos])
    model_motif = Model(inputs=[motif_inputs, motif_spatial_pos], outputs=[Outseq_motif, encoder_padding_mask_motif])

    motif_inputs1 = Input(shape=(None,), name="molecule_sequence1")
    motif_inputs2 = Input(shape=(None,), name="molecule_sequence2")
    spatial_pos1 = Input(shape=(None, None), dtype=tf.int32, name="spatial_pos1")
    spatial_pos2 = Input(shape=(None, None), dtype=tf.int32, name="spatial_pos2")
    druga_trans, _ = model_motif([motif_inputs1, spatial_pos1])
    drugb_trans, _ = model_motif([motif_inputs2, spatial_pos2])

    #Fingerprint
    smiles_input1 = Input(shape=(), dtype=tf.string, name="smiles1")
    smiles_input2 = Input(shape=(), dtype=tf.string, name="smiles2")
    default_cache_file = os.path.join(dataFolder, "fingerprint_cache.pkl")
    fp_extractor = MolFingerprintMLPExtractor(feature_dim=args.embedding_num, verbose=False, cache_file=default_cache_file)
    fp_feat_a = fp_extractor(smiles_input1)
    fp_feat_b = fp_extractor(smiles_input2)

    #R2AM_layer
    R2AM_layer = AsyMNFusion(
        feature_dim=args.embedding_num, stats_a_path=args.stats_a, stats_b_path=args.stats_b,
        mi_temperature=args.mi_temperature, local_temperature=args.local_temperature,
        responsibility_temperature=args.responsibility_temperature,
        redundancy_weight=args.redundancy_weight, name="R2AM_layer"
    )

    fusion_outputs = R2AM_layer([druga_trans, drugb_trans, kg_a, kg_b, kg_pair, fp_feat_a, fp_feat_b])
    pair_repr = fusion_outputs["pair_h"]

    #classifier, EDL output
    x = Dense(d_model, activation='relu', kernel_regularizer=l2(1e-7), name="cls_fc1")(pair_repr)
    x = Dropout(dropout_rate, name="cls_dropout1")(x)
    x = Dense(d_model // 2, activation='relu', kernel_regularizer=l2(1e-7), name="cls_fc2")(x)
    x = Dropout(dropout_rate, name="cls_dropout2")(x)
    logits_4 = Dense(4, activation=None, kernel_regularizer=l2(1e-7), name="output_logits")(x)
    output1_2 = DirichletAlphasFromLogits(name="dirichlet_alphas")(logits_4)

    models = Model(
        inputs=[motif_inputs1, motif_inputs2, spatial_pos1, spatial_pos2,
                input1_layer, input2_layer, smiles_input1, smiles_input2],
        outputs=[output1_2]
    )

    #compiling
    opt = Adam(learning_rate=args.learning_rate)
    loss_fn = DirichletEvidentialLoss(num_classes=args.event_num, lam=args.dirichlet_lam, name="dirichlet_evidential_loss")
    models.compile(optimizer=opt, loss=loss_fn, metrics=['sparse_categorical_accuracy'])

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, mode='min', verbose=1)
    earlystopping = EarlyStopping(monitor='val_loss', patience=args.early_stop_patience, min_delta=5e-4, mode='min', restore_best_weights=True)

    seed_dir = os.path.join(script_directory, f"seed{seed}-{MODEL_RUN_SUFFIX}_{args.save_dir_suffix}")
    save_model_callback = SaveModelWithMetrics(base_path=seed_dir, monitor='val_loss', verbose=2, save_best_only=False, mode='min')

    print("=" * 60)
    print(f"begin training seed={seed}: AsyMN-DDI")
    print(f"权重目录 {seed_dir}")
    print("=" * 60)

    #training
    history = models.fit(
        train_dataset, epochs=args.epochs, initial_epoch=args.initial_epoch,
        callbacks=[earlystopping, reduce_lr, save_model_callback],
        validation_data=val_dataset, verbose=2
    )

    # tst
    test_results = models.evaluate(test_dataset, verbose=2)
    print(f"Seed {seed} test results:", test_results)

    #save JSON
    val_losses = history.history.get('val_loss', [])
    val_accs = history.history.get('val_sparse_categorical_accuracy', [])
    best_epoch = int(np.argmin(val_losses)) if val_losses else -1
    best_val_loss = float(min(val_losses)) if val_losses else float('nan')
    best_val_acc = float(val_accs[best_epoch]) if val_accs and best_epoch >= 0 else float('nan')

    summary = {
        "seed": seed,
        "best_val_loss": best_val_loss,
        "best_val_sparse_categorical_accuracy": best_val_acc,
        "best_val_loss_epoch_1based": best_epoch + 1 if best_epoch >= 0 else None,
        "epochs_ran": len(val_losses),
        "test_loss": float(test_results[0]),
        "test_sparse_categorical_accuracy": float(test_results[1]),
    }
    summary_path = os.path.join(seed_dir, "best_results.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    return summary

if __name__ == "__main__":
    seeds_to_run = [args.only_seed] if args.only_seed is not None else list(TRAIN_SEEDS)
    all_summaries = []
    for s in seeds_to_run:
        print("\n" + "#" * 60 + f"\n# RUN seed={s}\n" + "#" * 60 + "\n")
        all_summaries.append(train_one_seed(s))

    summary_csv = os.path.join(script_directory, f"all_seeds_best_summary_{MODEL_RUN_SUFFIX}.csv")
    pd.DataFrame(all_summaries).to_csv(summary_csv, index=False)
    print(f"ALL {len(seeds_to_run)} seeds: {summary_csv}")