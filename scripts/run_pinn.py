"""Script for training and plotting the PINN model."""

import os

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from viz import create_animation, plot_snapshots


from project import (
    generate_training_data,
    load_config,
    predict_grid,
    train_pinn,
)


def main():
    cfg = load_config("config.yaml")

    #######################################################################
    # Oppgave 5.4: Start
    #######################################################################


    x, y, t, _, sensor_data = generate_training_data(cfg)
    print('Generert sensordata')

    pinn_params, losses = train_pinn(sensor_data, cfg)
    print('Trent modell')

    T_pred = predict_grid(pinn_params['nn'], x, y, t, cfg)
    print('Laga gitter')

    print("\nGenerating PINN visualizations...")
    plot_snapshots(
        x,
        y,
        t,
        T_pred,
        save_path="output/pinn/pinn_snapshots.png",
    )
    create_animation(
        x, y, t, T_pred, title="PINN", save_path="output/pinn/pinn_animation.gif"
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
    os.makedirs("output/pinn", exist_ok=True)
    plt.savefig("output/pinn/pinn_losses.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: output/pinn/pinn_losses.png")
    
    print('Finished')

    #######################################################################
    # Oppgave 5.4: Slutt
    #######################################################################


if __name__ == "__main__":
    print('startar')
    main()
