import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# --- 統計數據 ---
# Headpose: [roll, yaw, pitch]
hp_means_normal = np.array([0.5806377,   84.44447014,    2.04253652])
hp_stds_normal  = np.array([5.03031027,  17.84081543,    1.3047409])

hp_means_drowsy = np.array([0.162024746, 89.5216754, -40887.9198])
hp_stds_drowsy  = np.array([8.26481623,  19.1621249, 1223203.48])

# Gaze: [yaw, pitch]
gaze_means_normal = np.array([0.11928733, -0.2728983])
gaze_stds_normal  = np.array([0.5282137,   0.24933891])

gaze_means_drowsy = np.array([0.11064478, -0.5397062])
gaze_stds_drowsy  = np.array([0.5744506,   0.35776722])

# EAR: [left, right]
ear_means_normal = np.array([0.24370688, 0.24108328])
ear_stds_normal  = np.array([0.07883933, 0.08222673])

ear_means_drowsy = np.array([0.15657156, 0.15849431])
ear_stds_drowsy  = np.array([0.09159044, 0.09613300])

# 統一配色
colors = {'Normal':'tab:blue', 'Drowsy':'tab:orange'}

# === 繪圖設定 ===
fig = plt.figure(figsize=(15, 10))
gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1], hspace=0.5, wspace=0.3)

# (1) Headpose Roll
ax_roll = fig.add_subplot(gs[0, 0])
ax_roll.bar(
    ['Normal', 'Drowsy'],
    [hp_means_normal[0], hp_means_drowsy[0]],
    yerr=[hp_stds_normal[0], hp_stds_drowsy[0]],
    capsize=5,
    color=[colors['Normal'], colors['Drowsy']]
)
ax_roll.set_title('Headpose Roll μ±σ')
ax_roll.set_ylabel('Degrees')

# (2) Headpose Yaw
ax_yaw = fig.add_subplot(gs[0, 1])
ax_yaw.bar(
    ['Normal', 'Drowsy'],
    [hp_means_normal[1], hp_means_drowsy[1]],
    yerr=[hp_stds_normal[1], hp_stds_drowsy[1]],
    capsize=5,
    color=[colors['Normal'], colors['Drowsy']]
)
ax_yaw.set_title('Headpose Yaw μ±σ')
ax_yaw.set_ylabel('Degrees')

# (3) Headpose Pitch with broken axis
gs2 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0, 2],
                                       height_ratios=[1, 3], hspace=0.05)
ax_p_up = fig.add_subplot(gs2[0])
ax_p_down = fig.add_subplot(gs2[1], sharex=ax_p_up)

means_pitch = [hp_means_normal[2], hp_means_drowsy[2]]
stds_pitch  = [hp_stds_normal[2],  hp_stds_drowsy[2]]
labels = ['Normal', 'Drowsy']

# 上半部：極端數值區間
ax_p_up.bar(labels, means_pitch, yerr=stds_pitch, capsize=5,
            color=[colors['Normal'], colors['Drowsy']])
ax_p_up.set_ylim(-50000, 20000)
ax_p_up.spines['bottom'].set_visible(False)
ax_p_up.tick_params(labelbottom=False)

# 下半部：合理範圍
ax_p_down.bar(labels, means_pitch, yerr=stds_pitch, capsize=5,
              color=[colors['Normal'], colors['Drowsy']])
ax_p_down.set_ylim(-10, 10)
ax_p_down.spines['top'].set_visible(False)
ax_p_down.set_ylabel('Headpose Pitch (°)')

# 斷軸標示
d = .015
# 上圖斜線
ax_p_up.plot((-d, +d), (-d, +d), transform=ax_p_up.transAxes,
             color='k', clip_on=False)
ax_p_up.plot((1 - d, 1 + d), (-d, +d), transform=ax_p_up.transAxes,
             color='k', clip_on=False)
# 下圖斜線
ax_p_down.plot((-d, +d), (1 - d, 1 + d), transform=ax_p_down.transAxes,
               color='k', clip_on=False)
ax_p_down.plot((1 - d, 1 + d), (1 - d, 1 + d), transform=ax_p_down.transAxes,
               color='k', clip_on=False)

ax_p_down.set_xticks([0, 1])
ax_p_down.set_xticklabels(labels)
ax_p_up.set_title('Headpose Pitch μ±σ (Broken Axis)')

# (4) Gaze μ ± σ (跨兩欄)
ax_gaze = fig.add_subplot(gs[1, 0:2])
x = np.arange(2)
width = 0.35
ax_gaze.bar(
    x - width/2, gaze_means_normal, width,
    yerr=gaze_stds_normal, capsize=5,
    color=colors['Normal'], label='Normal'
)
ax_gaze.bar(
    x + width/2, gaze_means_drowsy, width,
    yerr=gaze_stds_drowsy, capsize=5,
    color=colors['Drowsy'], label='Drowsy'
)
ax_gaze.set_xticks(x)
ax_gaze.set_xticklabels(['Yaw', 'Pitch'])
ax_gaze.set_ylabel('Gaze (rad)')
ax_gaze.set_title('Gaze μ±σ')
ax_gaze.legend()

# (5) EAR μ ± σ
ax_ear = fig.add_subplot(gs[1, 2])
x = np.arange(2)
ax_ear.bar(
    x - width/2, ear_means_normal, width,
    yerr=ear_stds_normal, capsize=5,
    color=colors['Normal'], label='Normal'
)
ax_ear.bar(
    x + width/2, ear_means_drowsy, width,
    yerr=ear_stds_drowsy, capsize=5,
    color=colors['Drowsy'], label='Drowsy'
)
ax_ear.set_xticks(x)
ax_ear.set_xticklabels(['Left Eye', 'Right Eye'])
ax_ear.set_ylabel('Eye Aspect Ratio (EAR)')
ax_ear.set_title('EAR μ±σ')
ax_ear.legend()

plt.tight_layout()
plt.show()
