import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

plot_path = 'C:/Users/Matth/Documents/Leiden University/Project/Masters Project Main/plots/Random_forest/'
dir = 'C:/Users/Matth/Documents/Leiden University/Project/Masters Project Main/Paper_stuff/'
file = 'radii_MR_5443.csv'
path = dir + file

df = pd.read_csv(path)

print(df.columns)


def plot_2d_hist(CEPAM,true):
    fig = plt.figure(figsize=(6,4))
    mse = mean_squared_error(CEPAM, true)
    print(f'Mean Squared Error: {mse}')

    r2 = r2_score(CEPAM, true)
    print(f'R-squared: {r2}')
    print(CEPAM.min(), true.min(),CEPAM.max(), true.max())
    min_val = min(CEPAM.min(), true.min())
    max_val = max(CEPAM.max(), true.max())
    h = plt.hist2d(true, CEPAM, bins=100, cmap="viridis", cmax=100, density=False)
    fig.colorbar(h[3], label="Density")
    plt.plot([min_val, max_val], [min_val, max_val],'k--',lw=1,label='Perfect prediction')
    plt.title(f'Real vs predicted values for verification')
    image_path = plot_path+f'real_vs_pred_verification_MR.png'
    plt.text(
        0.05, 0.95, f'$R^2 = {r2:.3f}$',
        transform=plt.gca().transAxes,  # place relative to axes (0–1)
        fontsize=12,
        verticalalignment='top',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray')
    )
    plt.text(
        0.05, 0.85, f'$MSE = {mse:.3f}$',
        transform=plt.gca().transAxes,  # place relative to axes (0–1)
        fontsize=12,
        verticalalignment='top',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray')
    )
    plt.ylim(min_val,max_val)
    plt.xlim(min_val, max_val)
    plt.legend()
    plt.tight_layout()
    plt.savefig(image_path)
    plt.show()

def baumeister_verification(CEPAM, true, mass):
    delta_R = CEPAM - true
    delta_R_over_rp = delta_R / true
    mass_g = mass*8.68103e+28
    mass = mass_g/5.972e+27
    true_cm = true*6.378e+8
    density = mass_g/(4/3*np.pi*true_cm**3)

    fig, axes = plt.subplots(2, 2, figsize=(12, 5))
    axes = axes.flatten()

    # ---- Subplot 1: Histogram ----
    ax = axes[0]
    ax.hist(delta_R_over_rp, bins=100)

    left_border = np.percentile(delta_R_over_rp, 10)
    right_border = np.percentile(delta_R_over_rp, 90)
    median = np.percentile(delta_R_over_rp, 50)

    ax.axvspan(left_border, right_border, color='c', alpha=0.3, label='80% coverage')
    ax.axvline(median, color='c', linestyle='solid', linewidth=1, label='Median')

    ax.set_xticks(np.arange(-2, 2, 0.1), minor=True)
    ax.set_ylabel('Counts',fontsize=18)
    ax.set_xlabel(r'$\Delta R / R_p$',fontsize=18)

    # ---- Subplot 2: Placeholder ----
    ax = axes[1]

    sc = ax.scatter(x=density,y=delta_R_over_rp,alpha = 0.5,c=true,cmap="viridis")
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Radius", fontsize=14)

    ax.set_ylabel(r'$\Delta R/R_p$',fontsize=18)
    ax.set_xlabel(r'$\rho_{bulk}(g cm^{-3})$',fontsize=18)
    ax.set_xlim(0,11)

    # ---- Subplot 3: Placeholder ----
    ax = axes[2]

    ax.scatter(x=mass, y=delta_R_over_rp, alpha = 0.05)


    ax.set_ylabel(r'$\Delta R/R_p$',fontsize=18)
    ax.set_xlabel(r'$M_p (M_{\oplus})$',fontsize=18)

    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.9])

    # ---- Subplot 3: Placeholder ----
    ax = axes[3]

    ax.scatter(x=true, y=delta_R_over_rp, alpha=0.05)

    ax.set_ylabel(r'$\Delta R/R_p$', fontsize=18)
    ax.set_xlabel(r'$R_p (R_{\oplus})$', fontsize=18)

    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.9])

    plt.show()


df = df.dropna(subset=['CEPAM_radius'])
# baumeister_verification(df['CEPAM_radius'],df['QRF_radius'],df['mass'])

df = df[df.CEPAM_radius != 0]
df['CEPAM_radius'] = df['CEPAM_radius'].apply(lambda x: round(x, 2))
df = df[df.CEPAM_radius != 4.92]
# df = df[df.random_sampled == False]

# baumeister_verification(df['CEPAM_radius'],df['QRF_radius'],df['mass'])

plt.scatter(df['mass']*8.68103e+28/5.972e+27,df['CEPAM_radius'],alpha=0.05)
plt.xlabel('mass')
plt.ylabel('radius')
plt.yscale('log')
plt.xscale('log')
plt.axvline(4.5e0)
plt.axvline(1.2e2)

plt.show()

