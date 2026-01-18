import torch
import torchvision
import torchvision.transforms as transforms
from transformers import ViTForImageClassification, ViTImageProcessor
import numpy as np
import time

def get_CIFAR_loader():
    print("Downloading CIFAR-10 test dataset...")

    # CIFAR-10 uses 32x32 images, but ViT expects 224x224
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    testset = torchvision.datasets.CIFAR10(root='./data_raw', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                             shuffle=False, num_workers=2)
    print(f"Dataset downloaded. Test set size: {len(testset)} images")
    return testloader

def get_custom_loader(data_path='./data1'):
    print(f"Using custom dataset from {data_path}")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    testset = torchvision.datasets.ImageFolder(root=data_path, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                             shuffle=False, num_workers=2)
    print(f"Dataset loaded. Test set size: {len(testset)} images")
    return testloader


def load_pretrained_model():
    model_name = "edumunozsala/vit_base-224-in21k-ft-cifar10"
    model = ViTForImageClassification.from_pretrained(model_name)
    model.eval()
    print(f"Model loaded: {model_name}")
    return model

def run_inference(model, dataloader):
    """Runs inference on the dataset and returns predictions and labels."""
    print("\nRunning inference on test set...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    start_time = time.time()

    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            outputs = model(images)
            logits = outputs.logits
            _, predicted = torch.max(logits, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

            if (i + 1) % 50 == 0:
                print(f"Processed {(i + 1) * dataloader.batch_size} images...")

    total_time = time.time() - start_time
    print(f"Inference complete. Total time: {total_time:.2f}s")

    return all_preds, all_labels, total_time

def calculate_accuracy(preds, labels):
    """Calculates the accuracy of the model."""
    print("\n" + "="*60)
    print("ACCURACY RESULTS")
    print("="*60)

    preds = np.array(preds)
    labels = np.array(labels)

    correct = np.sum(preds == labels)
    total = len(labels)
    accuracy = correct / total

    print(f"Correct predictions: {correct}/{total}")
    print(f"Test Accuracy: {accuracy*100:.2f}%")
    print("="*60)

    return accuracy

def benchmark_inference_time(model, dataloader, num_batches=100):
    """Benchmarks the inference time per batch."""
    print("\n" + "="*60)
    print("INFERENCE TIME BENCHMARK")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    times = []
    batch_sizes = []

    with torch.no_grad():
        for i, (images, _) in enumerate(dataloader):
            if i >= num_batches:
                break

            images = images.to(device)
            batch_sizes.append(images.size(0))

            # Warm up GPU if using CUDA
            if i == 0 and torch.cuda.is_available():
                _ = model(images)
                torch.cuda.synchronize()

            start_time = time.time()
            _ = model(images)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            end_time = time.time()
            times.append(end_time - start_time)

    avg_time = np.mean(times)
    std_time = np.std(times)
    avg_batch_size = np.mean(batch_sizes)
    time_per_image = avg_time / avg_batch_size

    print(f"Batches benchmarked: {len(times)}")
    print(f"Average batch size: {avg_batch_size:.0f}")
    print(f"Average time per batch: {avg_time*1000:.2f} ms (±{std_time*1000:.2f} ms)")
    print(f"Average time per image: {time_per_image*1000:.2f} ms")
    print(f"Throughput: {1/time_per_image:.2f} images/second")
    print("="*60)

    return avg_time, time_per_image

if __name__ == '__main__':
    print("="*60)
    print("PRETRAINED VISION TRANSFORMER - CIFAR-10 BENCHMARK")
    print("="*60)

    model = load_pretrained_model()

    testloader = get_CIFAR_loader()
    predictions, labels, total_inference_time = run_inference(model, testloader)
    accuracy = calculate_accuracy(predictions, labels)
    avg_batch_time, avg_image_time = benchmark_inference_time(model, testloader)

    # testloader = download_dataset()
    # predictions, labels, total_inference_time = run_inference(model, testloader)
    # accuracy = calculate_accuracy(predictions, labels)
    # avg_batch_time, avg_image_time = benchmark_inference_time(model, testloader)

    # Final summary
    print("\n" + "="*60)
    print("FINAL BENCHMARK SUMMARY")
    print("="*60)
    print(f"Model: Vision Transformer (ViT-Base) - Pretrained on CIFAR-10")
    print(f"Dataset: CIFAR-10 Test Set (10,000 images)")
    print(f"Device: {'GPU (CUDA)' if torch.cuda.is_available() else 'CPU'}")
    print(f"\nAccuracy: {accuracy*100:.2f}%")
    print(f"Average inference time: {avg_image_time*1000:.2f} ms/image")
    print(f"Throughput: {1/avg_image_time:.2f} images/second")
    print("="*60)
