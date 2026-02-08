"""Script for training and plotting the NN model."""

import os

import matplotlib.pyplot as plt
import numpy as np
from viz import create_animation, plot_snapshots

from project import (
    generate_training_data,
    load_config,
    predict_grid,
    train_nn,
)


def main():
    cfg = load_config("config.yaml")

    #######################################################################
    # Oppgave 4.4: Start
    #######################################################################

    x, y, t, _, sensor_data = generate_training_data(cfg)

    params, losses = train_nn(sensor_data, cfg)

    T_pred = predict_grid(params, x, y, t, cfg)

    print("\nGenerating NN visualizations...")
    plot_snapshots(
        x,
        y,
        t,
        T_pred,
        save_path="output/nn/nn_snapshots.png",
    )
    create_animation(
        x, y, t, T_pred, title="NN", save_path="output/nn/nn_animation.gif"
    )

    print('Visualizing losses...')
    plt.figure()
    plt.plot(losses['total'], label='total')
    plt.plot(losses['data'], label='data')
    plt.plot(losses['ic'], label='ic')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    os.makedirs("output/nn", exist_ok=True)
    plt.savefig("output/nn/nn_losses.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: output/nn/nn_losses.png")
    
    print('Finished')
    #######################################################################
    # Oppgave 4.4: Slutt
    #######################################################################


if __name__ == "__main__":
    main()
