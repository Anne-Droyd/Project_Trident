import pandas as pd
import matplotlib.pyplot as plt

file = "C:/Users/Matth/Downloads/PS_2025.12.15_16.56.11.csv"

data = pd.read_csv(file,skiprows=96)
data = data[['pl_name',
            'pl_rade','pl_radeerr1','pl_radeerr2',
            'pl_bmasse','pl_bmasseerr1','pl_bmasseerr2',
            'pl_eqt','pl_eqterr1','pl_eqterr2',
            'st_teff','st_tefferr1','st_tefferr2',
            'st_met','st_meterr1','st_meterr2']]
data = data.groupby('pl_name', as_index = False).mean()
data = data[data['pl_bmasse'].notna()]
print(len(data))
data = data[data['pl_rade'].notna()]
print(len(data))

#might need to get a source for quoting these figures
bins = [0, 0.1, 0.5, 2, 10, 50, 5000]
labels = ["mercurians", "sub-earths", "earths", "super-Earths", "Neptunians", "Jovians"]

radius_bins = [0, 0.5, 1.25, 2.0, 4.0, 10.0, 50.0]
radius_labels = ["sub-earths","earth-sized","super-earths","sub-neptunes","neptunes","giants"]

data["mass_category"] = pd.cut(data["pl_bmasse"], bins=bins, labels=labels, include_lowest=True)
data["radius_category"] = pd.cut(data["pl_rade"],bins=radius_bins,labels=radius_labels,include_lowest=True)

mass_counts = data["mass_category"].value_counts().reindex(labels)
radius_counts = (data["radius_category"].value_counts().reindex(radius_labels))


# --- Compute percentages ---
percentages = mass_counts / mass_counts.sum() * 100
radius_percentages = radius_counts / radius_counts.sum() * 100

# --- Print counts and percentages ---
print(pd.DataFrame({"Count": mass_counts, "Percentage": percentages.round(2)}))
print(pd.DataFrame({"Count": radius_counts,"Percentage": radius_percentages.round(2)}))

# --- Plot bar chart with percentages ---
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# --- Mass plot ---
mass_counts.plot(kind="bar", ax=axes[0], color="skyblue", edgecolor="black")

axes[0].set_ylabel("Count")
axes[0].tick_params(axis="x", rotation=45)

for i, (count, pct) in enumerate(zip(mass_counts, percentages)):
    axes[0].text(i, count + 1, f"{pct:.1f}%", ha="center", va="bottom")

# --- Radius plot ---
radius_counts.plot(kind="bar", ax=axes[1], edgecolor="black")


axes[1].tick_params(axis="x", rotation=45)

for i, (count, pct) in enumerate(zip(radius_counts, radius_percentages)):
    axes[1].text(i, count + 1, f"{pct:.1f}%", ha="center", va="bottom")
axes[0].set_xlabel("")
axes[1].set_xlabel("")
plt.tight_layout()
plt.show()



plt.scatter(data['pl_bmasse'],data['pl_rade'])
plt.xlabel('mass')
plt.ylabel('radius')
plt.yscale('log')
plt.xscale('log')

plt.axvline(4.5e0)
plt.axvline(1.2e2)
plt.show()
