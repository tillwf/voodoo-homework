import click
from voodoo_homework.data.make_dataset import dataset
from voodoo_homework.features.build_features import build
from voodoo_homework.features.merge_features import merge
from voodoo_homework.models.train_model import train
from voodoo_homework.models.predict_model import predict

cli = click.CommandCollection(sources=[
    dataset,
    build,
    merge,
    predict,
    train,
])

if __name__ == '__main__':
    cli()
