import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from matplotlib import pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix
from tqdm import tqdm
import ctypes
from sklearn.model_selection import train_test_split

ctypes.CDLL('/home/ff/anaconda3/envs/spike/lib/python3.9/site-packages/torch/lib/libtorch_cpu.so',
            mode=ctypes.RTLD_GLOBAL)
ctypes.CDLL(
    '/home/ff/anaconda3/envs/spike/lib/python3.9/site-packages/torch/lib/libtorch_python.so')
ctypes.CDLL(
    '/home/ff/anaconda3/envs/spike/lib/python3.9/site-packages/AsynSpiGCN.cpython-39-x86_64-linux-gnu.so',
    mode=ctypes.RTLD_GLOBAL)
import AsynSpiGCN
from torch_geometric.datasets import Amazon, Planetoid

device = torch.device("cpu")


def create_log_dir(log_dir):
    if not os.path.exists(log_dir):
        try:
            os.makedirs(log_dir)
            print(f"Created log directory: {log_dir}")
        except Exception as e:
            print(f"Error creating log directory {log_dir}: {e}")


def select_teacher_model(model_type, gcn_num_features, gcn_hidden_channels):
    if model_type == 'GCN':
        return GCNConv(gcn_num_features, gcn_hidden_channels)
    elif model_type == 'GAT':
        # GAT with 8 heads, output size becomes gcn_hidden_channels * 8 due to concat=True
        return GATConv(gcn_num_features, gcn_hidden_channels // 8, heads=8, concat=True)
    elif model_type == 'GraphSAGE':
        return SAGEConv(gcn_num_features, gcn_hidden_channels)
    else:
        raise ValueError(f"Invalid model type: {model_type}. Choose 'GCN', 'GAT', or 'GraphSAGE'.")


# Save metrics to CSV file
def save_metrics_to_csv(train_losses, test_losses, train_accuracies, test_accuracies, model_name, dataset_name,
                        output_dir):
    metrics_df = pd.DataFrame({
        'Epoch': range(1, len(train_losses) + 1),
        'Train_Loss': train_losses,
        'Test_Loss': test_losses,
        'Train_Accuracy': train_accuracies,
        'Test_Accuracy': test_accuracies
    })
    # Ensure directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, f'{model_name}_{dataset_name}_metrics.csv')
    metrics_df.to_csv(output_path, index=False)
    print(f"Metrics saved to {output_path}")


def calculate_similarity(reconstructed, original):
    return F.cosine_similarity(
        reconstructed, original.view(reconstructed.size(0), -1), dim=1).float().mean().item()


def standardize_features(input_tensor):
    # 计算每个特征的最大值和最小值
    min_val = input_tensor.min(dim=0, keepdim=True).values
    max_val = input_tensor.max(dim=0, keepdim=True).values
    # 最小-最大归一化，加入小的常数1e-6防止除以零
    normalized = (input_tensor - min_val) / (max_val - min_val + 1e-6)
    return normalized


def reparameterize_with_bounds(mu, logvar, lower_bound, upper_bound):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    z = mu + eps * std
    z = lower_bound + (upper_bound - lower_bound) * torch.sigmoid(z)
    return z


class SGCN_Module(nn.Module):
    def __init__(self, v_rest,
                 v_r, sample_num, speed_init, threshold_init, refractory_period_init, decay_alpha_init,
                 time_step_init, max_rate_init, beta_init, fr_threshold, input_dim, hidden_dim, parallel_batch):
        super(SGCN_Module, self).__init__()
        self.SGCN_lin = nn.Linear(input_dim, hidden_dim)
        self.SGCN = AsynSpiGCN.SGCN(v_rest,
                                    v_r,
                                    sample_num,
                                    speed_init,
                                    threshold_init,
                                    refractory_period_init,
                                    decay_alpha_init,
                                    time_step_init,
                                    max_rate_init,
                                    beta_init,
                                    fr_threshold)
        self.parallel_batch = parallel_batch
        self.hidden_dim = hidden_dim
        self.spike_node = None

    def forward(self, data):
        x = self.SGCN_lin(data.x)
        x = standardize_features(x)
        self.SGCN.init_param(data.node_indices, self.hidden_dim)
        fusion_tensor, spike_matrix = self.SGCN.forward(data.spike_node, x, data.edge_index,
                                                        data.node_indices, data.edge_attr, self.parallel_batch)
        self.spike_node = self.SGCN.get_spike_nodes()
        return fusion_tensor, spike_matrix


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Linear(in_channels, out_channels)
            )

    def forward(self, x):
        out = self.fc(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Spi_VAE_Module(nn.Module):
    def __init__(self, m, n, latent_dim=64):  # 增大潜在空间尺度
        super(Spi_VAE_Module, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(m * n, 1024),  # 增加神经元数量
            nn.ReLU(),
            nn.Dropout(0.3),
            ResidualBlock(1024, 512),  # 增加残差块
            nn.Dropout(0.3),
            ResidualBlock(512, 256),
            nn.Dropout(0.3),
            nn.Linear(256, latent_dim),  # 增加潜在空间维度
            nn.ReLU()
        )

        # 分别用于两个分布的均值和对数方差
        self.fc_mu_0 = nn.Linear(latent_dim, m * n)
        self.fc_logvar_0 = nn.Linear(latent_dim, m * n)
        self.fc_mu_1 = nn.Linear(latent_dim, m * n)
        self.fc_logvar_1 = nn.Linear(latent_dim, m * n)

        # 解码部分也增加残差连接
        self.decoder = nn.Sequential(
            nn.Linear(m * n, 256),
            nn.ReLU(),
            ResidualBlock(256, 512),
            nn.ReLU(),
            ResidualBlock(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, m * n),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(1, -1)
        h = self.encoder(x)
        mu_0, logvar_0 = self.fc_mu_0(h), self.fc_logvar_0(h)
        mu_1, logvar_1 = self.fc_mu_1(h), self.fc_logvar_1(h)
        # Sample from the distributions with the desired bounds
        z_0 = reparameterize_with_bounds(
            mu_0, logvar_0, lower_bound=0.5, upper_bound=1)
        z_1 = reparameterize_with_bounds(
            mu_1, logvar_1, lower_bound=0, upper_bound=0.5)
        reconstructed = self.decoder(torch.where(x >= 0.5, z_1, z_0))
        return reconstructed, mu_0, logvar_0, mu_1, logvar_1


class Graph_Net(nn.Module):
    def __init__(self,
                 v_rest,
                 v_r,
                 sample_num,
                 speed_init,
                 threshold_init,
                 refractory_period_init,
                 decay_alpha_init,
                 time_step_init,
                 max_rate_init,
                 beta_init,
                 fr_threshold,
                 lin_input_dim,
                 lin_hidden_dim,
                 gcn_num_features,
                 gcn_hidden_channels,
                 m, n, latent_dim,
                 num_classes,
                 parallel_batch
                 ):
        super(Graph_Net, self).__init__()
        self.gcn_teacher1 = GCNConv(gcn_num_features, gcn_hidden_channels[0])
        self.gcn_teacher2 = GCNConv(gcn_hidden_channels[0], gcn_hidden_channels[1])
        self.sgcn_student1 = SGCN_Module(v_rest,
                                         v_r,
                                         sample_num,
                                         speed_init,
                                         threshold_init,
                                         refractory_period_init,
                                         decay_alpha_init,
                                         time_step_init,
                                         max_rate_init,
                                         beta_init,
                                         fr_threshold,
                                         lin_input_dim,
                                         lin_hidden_dim[0],
                                         parallel_batch)
        self.sgcn_student2 = SGCN_Module(v_rest,
                                         v_r,
                                         sample_num,
                                         speed_init,
                                         threshold_init,
                                         refractory_period_init,
                                         decay_alpha_init,
                                         time_step_init,
                                         max_rate_init,
                                         beta_init,
                                         fr_threshold,
                                         lin_hidden_dim[0],
                                         lin_hidden_dim[1],
                                         parallel_batch)
        self.spi_vae1 = Spi_VAE_Module(lin_hidden_dim[0], n, latent_dim)
        self.spi_vae2 = Spi_VAE_Module(lin_hidden_dim[1], n, latent_dim)
        self.classifier1 = nn.Linear(lin_hidden_dim[0], num_classes)
        self.classifier2 = nn.Linear(lin_hidden_dim[1], num_classes)
        self.fusion_layer1 = nn.Linear(2, lin_hidden_dim[0])
        self.fusion_transfer1 = nn.Sequential(
            nn.Linear(lin_hidden_dim[0], lin_hidden_dim[0]),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.fusion_layer2 = nn.Linear(2, lin_hidden_dim[1])
        self.fusion_transfer2 = nn.Sequential(
            nn.Linear(lin_hidden_dim[1], lin_hidden_dim[1]),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, data):
        spike_num = []
        fusion_tensor1, spike_matrix1 = self.sgcn_student1(data)
        data.spike_node = self.sgcn_student1.spike_node
        gcn_matrix1 = self.gcn_teacher1(data.x, data.edge_index)
        fusion_matrix1 = self.fusion_layer1(fusion_tensor1)
        if data.spike_node is None:
            spike_num.append(0)
            mapped_snn_matrix1, mu_0, logvar_0, mu_1, logvar_1 = self.spi_vae1(spike_matrix1)
            mapped_snn_matrix1 = mapped_snn_matrix1.view(fusion_matrix1.shape[0], fusion_matrix1.shape[1])
            fused_features1 = mapped_snn_matrix1 + fusion_matrix1
            fused_transfered_features = self.fusion_transfer1(fused_features1)
            out = self.classifier1(fused_transfered_features)
            return gcn_matrix1, out, fused_features1, mu_0, logvar_0, mu_1, logvar_1, spike_num
        else:
            spike_num.append(len(self.sgcn_student1.spike_node))
            fused_features1 = spike_matrix1 + fusion_matrix1
            fused_transfered_features1 = self.fusion_transfer1(fused_features1)
            data.x = fused_transfered_features1
            x = F.relu(gcn_matrix1)
            fusion_tensor2, spike_matrix2 = self.sgcn_student2(data)
            if self.sgcn_student2.spike_node is not None:
                spike_num.append(len(self.sgcn_student2.spike_node))
            else:
                spike_num.append(0)
            gcn_matrix2 = self.gcn_teacher2(x, data.edge_index)
            mapped_snn_matrix, mu_0, logvar_0, mu_1, logvar_1 = self.spi_vae2(
                spike_matrix2)
            fusion_matrix2 = self.fusion_layer2(fusion_tensor2)
            mapped_snn_matrix = mapped_snn_matrix.view(
                fusion_matrix2.shape[0], fusion_matrix2.shape[1])
            fused_features2 = mapped_snn_matrix + fusion_matrix2
            fused_transfered_features2 = self.fusion_transfer2(fused_features2)
            out = self.classifier2(fused_transfered_features2)
            return gcn_matrix2, out, fused_features2, mu_0, logvar_0, mu_1, logvar_1, spike_num


class CustomLoss(nn.Module):
    def __init__(self, lambda_vae=1.0):
        super(CustomLoss, self).__init__()
        self.classification_loss = nn.CrossEntropyLoss()
        self.lambda_vae = lambda_vae

    def forward(self, gcn_output, model_output, target_labels, fused_features, mu_0,
                logvar_0, mu_1, logvar_1):
        # 1. 分类损失
        loss_cls = self.classification_loss(model_output, target_labels)

        # 3. VAE损失：包括重构误差和KL散度损失
        # 3.1 重构误差 (MSE between reconstructed and original spike matrix)
        mse_loss = F.mse_loss(fused_features, gcn_output.view(
            gcn_output.size(0), -1), reduction='sum')

        # 3.2 KL散度损失
        kld_0 = -0.5 * torch.sum(1 + logvar_0 - mu_0.pow(2) - logvar_0.exp())
        kld_1 = -0.5 * torch.sum(1 + logvar_1 - mu_1.pow(2) - logvar_1.exp())

        # 3.3 总VAE损失
        loss_vae = mse_loss + kld_0 + kld_1

        # 4. 总损失：分类损失 + 融合损失 + VAE损失
        total_loss = (1 - self.lambda_vae) * loss_cls + self.lambda_vae * loss_vae

        return total_loss


loss_func = CustomLoss(lambda_vae=0.8)


def train(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    gcn_matrix_train, out, fused_features_train, mu_0_train, logvar_0_train, mu_1_train, logvar_1_train, spike_num = model(
        data)
    loss_train = loss_func(gcn_matrix_train, out[data.train_mask], data.y[data.train_mask], fused_features_train,
                           mu_0_train, logvar_0_train, mu_1_train, logvar_1_train)
    label_train = data.y[data.train_mask]
    similarity = calculate_similarity(fused_features_train, gcn_matrix_train)
    # 计算准确率
    _, predicted_labels = torch.max(out[data.train_mask], 1)
    correct_predictions = (predicted_labels == label_train).sum().item()
    accuracy = correct_predictions / label_train.size(0)
    loss_train.backward()
    optimizer.step()
    return loss_train.item(), accuracy, similarity, spike_num


def test(model, data):
    model.eval()
    gcn_matrix_test, out, fused_features_test, mu_0_test, logvar_0_test, mu_1_test, logvar_1_test, spike_num = model(
        data)
    loss_test = loss_func(gcn_matrix_test, out[data.test_mask], data.y[data.test_mask], fused_features_test,
                          mu_0_test, logvar_0_test, mu_1_test, logvar_1_test)
    label_test = data.y[data.test_mask]
    similarity = calculate_similarity(fused_features_test, gcn_matrix_test)
    # 计算准确率
    _, predicted_labels = torch.max(out[data.test_mask], 1)
    correct_predictions = (predicted_labels == label_test).sum().item()
    accuracy = correct_predictions / label_test.size(0)
    return loss_test.item(), accuracy, similarity, data.y[data.test_mask].cpu().numpy(), out[data.test_mask].max(1)[
        1].cpu().numpy(), spike_num


def graph_construct(graph):
    graph.edge_attr = torch.rand(graph.edge_index.size(1))
    graph.node_indices = torch.arange(graph.num_nodes, dtype=torch.int32).to(device=device)
    graph.spike_node = graph.node_indices
    return graph


def save_classification_report(labels, pred, model_name, dataset_name, output_dir):
    report = classification_report(labels, pred)
    output_path = os.path.join(output_dir, f'{model_name}_{dataset_name}_classification_report.txt')
    with open(output_path, 'w') as file:
        file.write("Classification Report:\n")
        file.write(report)
    print(f"Classification report saved to {output_path}")


def plot_multiclass_roc(labels, pred, num_classes, model_name, dataset_name, output_dir):
    # Binarize the labels
    labels = label_binarize(labels, classes=range(num_classes))
    pred = label_binarize(pred, classes=range(num_classes))

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(labels[:, i], pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(labels.ravel(), pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"], label='micro-average ROC curve (area = {0:0.2f})'
                                               ''.format(roc_auc["micro"]), color='deeppink', linestyle=':',
             linewidth=4)

    for i in range(num_classes):
        plt.plot(fpr[i], tpr[i], lw=2, label='ROC curve of class {0} (area = {1:0.2f})'.format(
            i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic for {model_name} on {dataset_name}')
    plt.legend(loc="lower right")

    # Save the plot
    plot_path = os.path.join(output_dir, f'{model_name}_{dataset_name}_roc.png')
    plt.savefig(plot_path)
    print(f'ROC plot saved at {plot_path}')
    plt.clf()


def plot_confusion_matrix(labels, pred, model_name, dataset_name, output_dir, show_plot=False):
    # Compute confusion matrix
    cm = confusion_matrix(labels, pred)
    num_classes = cm.shape[0]

    # Create the plot
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Reds)
    plt.title(f'Confusion Matrix for {model_name} on {dataset_name}')
    plt.colorbar()

    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, np.arange(num_classes), rotation=45)
    plt.yticks(tick_marks, np.arange(num_classes))

    # Add labels to the plot
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    plt.tight_layout()

    # Save the plot
    plot_path = os.path.join(output_dir, f'{model_name}_{dataset_name}_confusion_matrix.png')
    plt.savefig(plot_path)
    print(f'Confusion matrix plot saved at {plot_path}')

    # Show the plot if requested
    if show_plot:
        plt.show()

    plt.clf()  # Clear the current figure


def plot_metrics(train_losses, test_losses, train_accuracies, test_accuracies, train_similarity, test_similarity,
                 model_name, dataset_name, output_dir):
    # Create output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(18, 5))

    # Training and Testing Loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_losses, label='Train Loss', color='blue')
    plt.plot(epochs, test_losses, label='Test Loss', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()

    # Training and Testing Accuracy
    plt.subplot(1, 3, 2)
    plt.plot(epochs, train_accuracies, label='Train Accuracy', color='blue')
    plt.plot(epochs, test_accuracies, label='Test Accuracy', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Epochs')
    plt.legend()

    # Training and Testing Similarity
    plt.subplot(1, 3, 3)
    plt.plot(epochs, train_similarity, label='Train Similarity', color='blue')
    plt.plot(epochs, test_similarity, label='Test Similarity', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Similarity')
    plt.title('Similarity over Epochs')
    plt.legend()

    # Save the plot
    plot_path = os.path.join(output_dir, f'{model_name}_{dataset_name}_metrics_over_epochs.png')
    plt.savefig(plot_path)
    print(f'Metrics plot saved in directory: {plot_path}')
    plt.clf()


def plot_threshold_scatters(thresholds, model_name, dataset_name, output_dir, num):
    # Create output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Generate scatter plot for V_thresholds
    if isinstance(thresholds, torch.Tensor):
        thresholds = thresholds.detach().numpy()

    epochs = list(range(len(thresholds)))  # X-axis: Epochs

    plt.scatter(epochs, thresholds, c='red', marker='*')
    plt.title(f'Scatter Plot of V_thresholds for {model_name} on {dataset_name}')
    plt.xlabel('Epochs')
    plt.ylabel('V_thresholds')

    # Save the plot
    plot_path = os.path.join(output_dir, f'{model_name}_{dataset_name}_v_thresholds{num}.png')
    plt.savefig(plot_path)
    print(f'V_thresholds scatter plot saved in directory: {plot_path}')
    plt.clf()


def main_experiment():
    datasets = {
        # 'CiteSeer': '/home/ff/code/spike/node_classification/data/CiteSeer',
        # 'Computers': '/home/ff/code/spike/node_classification/data',
        # /home/ff/code/spike/node_classification/data/Photo
        'Amazon-Photo': '/home/ff/code/spike/node_classification/data',
        'Cora': '/home/ff/code/spike/node_classification/data'
    }
    teacher_models = ['GCN', 'GraphSAGE']

    for dataset_name, dataset_root in datasets.items():
        # Load the dataset
        if dataset_name == 'CiteSeer':
            dataset = Planetoid(root=dataset_root, name='CiteSeer')
        elif dataset_name == 'Cora':
            dataset = Planetoid(root=dataset_root, name='Cora')
        elif dataset_name == 'Amazon-Photo':
            dataset = Amazon(root=dataset_root, name='Photo')
        elif dataset_name == 'Computers':
            dataset = Amazon(root=dataset_root, name='Computers')

        data = dataset[0].to(device)
        data = graph_construct(data)

        num_nodes = data.num_nodes  # 获取节点数
        num_classes = dataset.num_classes  # 获取类别数

        num_epochs = 2000
        v_rest = 0.0
        v_r = 0.0
        threshold_init = 1.0
        sample_num = 5
        speed_init = 1e-10
        refractory_period_init = 1e9
        decay_alpha_init = 0.1
        time_step_init = 1
        beta_init = 0.5
        fr_threshold = 0.5
        lin_hidden_dims = [256, 512]
        parallel_batch = 10
        lin_input_dim = data.x.shape[1]
        gcn_num_features = data.x.shape[1]
        m = 512
        gcn_hidden_channels = [256, 512]
        max_rate_init = 1.0
        n = data.x.shape[0]
        latent_dim = 64
        num_classes = dataset.num_classes
        labels = data.y
        train_idx, test_idx = train_test_split(torch.arange(num_nodes), test_size=0.2, stratify=labels)
        train_idx, val_idx = train_test_split(train_idx, test_size=0.2, stratify=labels[train_idx])

        data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.train_mask[train_idx] = True
        data.val_mask[val_idx] = True
        data.test_mask[test_idx] = True

        for model_type in teacher_models:
            # Initialize teacher model
            teacher_model1 = select_teacher_model(model_type, gcn_num_features, gcn_hidden_channels[0])
            teacher_model2 = select_teacher_model(model_type, gcn_hidden_channels[0], gcn_hidden_channels[1])

            # Initialize the main model with the selected teacher model
            model = Graph_Net(v_rest, v_r, sample_num, speed_init, threshold_init, refractory_period_init,
                              decay_alpha_init,
                              time_step_init, max_rate_init, beta_init, fr_threshold, lin_input_dim, lin_hidden_dims,
                              gcn_num_features, gcn_hidden_channels, m, n, latent_dim, num_classes, parallel_batch)
            model.gcn_teacher1 = teacher_model1.to(device)
            model.gcn_teacher2 = teacher_model2.to(device)
            optimizer = torch.optim.Adam([
                {'params': model.sgcn_student1.SGCN.parameters(), 'lr': 0.0001,
                 'weight_decay': 0.01},
                {'params': model.sgcn_student2.SGCN.parameters(), 'lr': 0.0001,
                 'weight_decay': 0.01},
                {'params': model.parameters(), 'lr': 0.0001, 'weight_decay': 0.001}
            ])

            # Set up SummaryWriter for TensorBoard
            log_dir = f'/home/ff/code/spike/tensorboard_logs/{dataset_name}/{model_type}_2Layers_conduct_last'
            create_log_dir(log_dir)  # Ensure the log directory exists
            writer = SummaryWriter(log_dir=log_dir)

            # Create output directory for this model and dataset
            output_dir = f'/home/ff/code/spike/node_classification/results_{dataset_name}_{model_type}_2Layers_conduct_last'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Training and Testing
            train_losses, test_losses = [], []
            train_accuracies, test_accuracies = [], []
            train_similarities, test_similarities = [], []
            thresholds1 = []
            thresholds2 = []
            x = data.x
            node_indices = data.node_indices
            for epoch in tqdm(range(num_epochs), desc=f"Training {model_type} on {dataset_name}", unit="epoch"):
                train_loss, train_acc, train_similarity, spike_num_train = train(model, data, optimizer)
                data.x = x
                data.spike_node = node_indices
                test_loss, test_acc, test_similarity, labels, pred, spike_num_test = test(model, data)
                data.x = x
                data.spike_node = node_indices
                # Log metrics
                train_losses.append(train_loss)
                test_losses.append(test_loss)
                train_accuracies.append(train_acc)
                test_accuracies.append(test_acc)
                train_similarities.append(train_similarity)
                test_similarities.append(test_similarity)

                writer.add_scalar('Loss/Train', train_loss, epoch)
                writer.add_scalar('Loss/Test', test_loss, epoch)
                writer.add_scalar('Accuracy/Train', train_acc, epoch)
                writer.add_scalar('Accuracy/Test', test_acc, epoch)
                writer.add_scalar('Similarity/Train', train_similarity, epoch)
                writer.add_scalar('Similarity/Test', test_similarity, epoch)
                writer.add_scalar('Spike_num_layer1/Train', spike_num_train[0], epoch)
                writer.add_scalar('Spike_num_layer1/Test', spike_num_test[0], epoch)
                if len(spike_num_train) == 2:
                    writer.add_scalar('Spike_num_layer2/Train', spike_num_train[1], epoch)
                if len(spike_num_test) == 2:
                    writer.add_scalar('Spike_num_layer2/Train', spike_num_train[1], epoch)

                # 更新进度条后面的描述信息
                tqdm.write(
                    f"Epoch {epoch + 1}/{num_epochs}:\n"
                    f"\tTrain-Loss: {train_loss:.4f}, Train-Accuracy: {train_acc:.4f}, Train-Similarity:{train_similarity:.4f}\n"
                    f"\tTest-Loss: {test_loss:.4f}, Test-Accuracy: {test_acc:.4f}, Test-Similarity:{test_similarity:.4f}",
                )

                vTreshold1 = model.sgcn_student1.SGCN.get_threshold()
                thresholds1.append(vTreshold1.item())
                vTreshold2 = model.sgcn_student2.SGCN.get_threshold()
                thresholds2.append(vTreshold2.item())

            writer.close()

            save_metrics_to_csv(
                train_losses, test_losses, train_accuracies, test_accuracies, model_type, dataset_name, output_dir
            )

            # Save classification report to text file
            save_classification_report(labels, pred, model_type, dataset_name, output_dir)

            # Plot ROC, Confusion Matrix, Metrics, and V_thresholds
            plot_multiclass_roc(labels, pred, num_classes, model_type, dataset_name, output_dir)
            plot_confusion_matrix(labels, pred, model_type, dataset_name, output_dir)
            plot_metrics(train_losses, test_losses, train_accuracies, test_accuracies, train_similarities,
                         test_similarities, model_name=model_type, dataset_name=dataset_name, output_dir=output_dir)
            plot_threshold_scatters(thresholds1, model_type, dataset_name, output_dir, 1)
            plot_threshold_scatters(thresholds2, model_type, dataset_name, output_dir, 2)


if __name__ == '__main__':
    main_experiment()
