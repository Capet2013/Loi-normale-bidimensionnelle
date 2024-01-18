import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

##############
# Question 1 #
##############

# Définition de la loi
mu = np.array([1, 2])
cov = np.array([[0.25, 0.3], [0.3, 0.75]])

def fz(z):
    return (1 / (np.sqrt((2 * np.pi) ** 2 * np.linalg.det(cov)))) * np.exp((-0.5) * (z - mu).dot(np.linalg.inv(cov)).dot((z - mu).T))

# Échantillon de points
samples = np.random.multivariate_normal(mu, cov, size=1000)
xmin = np.min(samples[:, 0]) - 0.5
xmax = np.max(samples[:, 0]) + 0.5
ymin = np.min(samples[:, 1]) - 0.5
ymax = np.max(samples[:, 1]) + 0.5
xscale = np.arange(xmin, xmax, 0.01) 
yscale = np.arange(ymin, ymax, 0.01) 

# Calcul des densités de probabilité
z = np.zeros((len(yscale), len(xscale)))
for i, x in enumerate(xscale):
    for j, y in enumerate(yscale):
        z[j, i] = fz(np.array([x, y]))

# Définition d'une ellipse
def ellipse(p, color):
    plt.contour(xscale, yscale, z, levels=[(1 - p) / np.sqrt((2 * np.pi) ** 2 * np.linalg.det(cov))], colors=color, linewidths=0.5)

# Affichage des points et des ellipses pour p = 0.99 / O.9 / 0.5 / 0.1
plt.figure(num='Question 1')
plt.scatter(samples[:, 0], samples[:, 1], alpha=0.3)
ellipse(0.99, 'blue')
ellipse(0.9, 'green')
ellipse(0.5, 'red')
ellipse(0.1, 'orange')

# Création de la légende pour les ellipses
legend_patches = [
    mpatches.Patch(color='blue', label='p = 0.99'),
    mpatches.Patch(color='green', label='p = 0.9'),
    mpatches.Patch(color='red', label='p = 0.5'),
    mpatches.Patch(color='orange', label='p = 0.1')
]
plt.legend(handles=legend_patches)

plt.title("Échantillon de points et ellipses d'isodensité")

##############
# Question 2 #
##############

# Fonction densité enstimée
def fz_estimated(z, mu_estimated, cov_estimated):
    return (1 / (np.sqrt((2 * np.pi) ** 2 * np.linalg.det(cov_estimated)))) * np.exp((-0.5) * (z - mu_estimated).dot(np.linalg.inv(cov_estimated)).dot((z - mu_estimated).T))

# Définition d'une ellipse en fonction de l'échantillon et de la covariance 
def ellipse2(ax, x, y, z1, z2, p, color, cov1, cov2):

    # Ellipse avec le z réel et la covariance donnée
    ax.contour(x, y, z1, levels=[(1 - p) / np.sqrt((2 * np.pi) ** 2 * np.linalg.det(cov1))], colors=color, linewidths=0.5)

    # Ellipse avec le z estimé et la covariance estimée
    contour = ax.contour(x, y, z2, levels=[(1 - p) / np.sqrt((2 * np.pi) ** 2 * np.linalg.det(cov2))], colors=color, linewidths=0.5)
    for contour_path in contour.collections:
        contour_path.set_linestyle('dashed')

    # Création de 2 lignes invisibles pour pouvoir faire la légende
    ax.plot([], [], color=color, linewidth=0.5, linestyle='dashed', label='Estimation')
    ax.plot([], [], color=color, linewidth=0.5, linestyle='solid', label='Réelle')
    
# Création de la fenêtre
fig, axes = plt.subplots(2, 2, figsize=(8, 8),)

# Boucle pour chaque sous-graphique
for i, ax in enumerate(axes.flatten()):

    # Définition des différents échantillons de points
    n_points = [50, 200, 1000, 5000][i]

    # Calcul des points
    samples = np.random.multivariate_normal(mu, cov, n_points)
    xmin = np.min(samples[:, 0]) - 0.5
    xmax = np.max(samples[:, 0]) + 0.5
    ymin = np.min(samples[:, 1]) - 0.5
    ymax = np.max(samples[:, 1]) + 0.5
    xscale = np.arange(xmin, xmax, 0.01)
    yscale = np.arange(ymin, ymax, 0.01)

    # Estimation des paramètres de la loi à partir des points
    mu_estimated = np.mean(samples, axis=0)
    cov_estimated = np.cov(samples.transpose())

    # Calcul des densités de probabilités
    z_real = np.zeros((len(yscale), len(xscale)))
    z_estimated = np.zeros((len(yscale), len(xscale)))
    for i, x in enumerate(xscale):
        for j, y in enumerate(yscale):
            z_real[j, i] = fz(np.array([x, y]))
            z_estimated[j, i] = fz_estimated(np.array([x, y]), mu_estimated, cov_estimated)

    # Affichage des points et des ellipses pour p = 0.1 / 0.99
    ax.scatter(samples[:, 0], samples[:, 1], alpha=0.3)
    ellipse2(ax, xscale, yscale, z_real, z_estimated, 0.1, 'blue', cov, cov_estimated)
    ellipse2(ax, xscale, yscale, z_real, z_estimated, 0.99, 'red', cov, cov_estimated)
    ax.set_title(f'Échantillon de taille {n_points}')

# Affichage de la légende, du titre et des 2 fenêtres
axes[0, 0].legend(loc='upper left')
plt.suptitle("Convergence des estimateurs μ et Σ")
plt.show()