import loader as Loader
import trainer as Trainer
import classifier

f_cuda = False
Loader.f_cuda = f_cuda
classifier.f_cuda = f_cuda
Trainer.f_cuda = f_cuda
def main():
    loader = Loader.Loader()
    loader.createDataset()
    trainer = Trainer.Training(loader.train_loader, loader.test_loader, loader.valida_folder)
    trainer.procedure()


if __name__ == "__main__":
    main()