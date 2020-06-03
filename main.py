# from loader import Loader
# from trainer import Training
from test import Training
def main():
    trainer = Training()
    trainer.createDataset()
    trainer.Procedure()
    # loader = Loader()
    # loader.createDataset()
    # trainer = Training(loader.train_loader, loader.test_loader, loader.valida_folder)
    # trainer.procedure()


if __name__ == "__main__":
    main()