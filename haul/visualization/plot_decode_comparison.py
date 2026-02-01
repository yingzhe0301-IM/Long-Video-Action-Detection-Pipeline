import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取数据
df_old = pd.read_csv('haul/haul_scan_merged_decode_all.csv')
df_eff = pd.read_csv('haul/haul_scan_merged_decode_on_demand.csv')

powers_of_2 = [1, 2, 4, 8, 16, 32, 64]

# === 图1: Comparison (4条线) ===
fig, ax = plt.subplots(figsize=(10, 6))

# Old decode
total_old_no = df_old['runtime_no_prefetch'].sum() / 3600
total_old_pf = df_old['runtime_prefetch'].sum() / 3600
ax.plot(df_old['frame_skip'], df_old['runtime_no_prefetch']/60,
        'o-', color='#e74c3c', linewidth=2, markersize=4, alpha=0.7,
        label=f'Decode All, No Prefetch (~{total_old_no:.1f}h)')
ax.plot(df_old['frame_skip'], df_old['runtime_prefetch']/60,
        's-', color='#c0392b', linewidth=2, markersize=4,
        label=f'Decode All, Prefetch (~{total_old_pf:.1f}h)')

# Efficient decode
total_eff_no = df_eff['runtime_no_prefetch'].sum() / 3600
total_eff_pf = df_eff['runtime_prefetch'].sum() / 3600
ax.plot(df_eff['frame_skip'], df_eff['runtime_no_prefetch']/60,
        '^-', color='#3498db', linewidth=2, markersize=4, alpha=0.7,
        label=f'Decode On Demand, No Prefetch (~{total_eff_no:.1f}h)')
ax.plot(df_eff['frame_skip'], df_eff['runtime_prefetch']/60,
        'd-', color='#2980b9', linewidth=2, markersize=4,
        label=f'Decode On Demand, Prefetch (~{total_eff_pf:.1f}h)')

ax.set_xscale('log', base=2)
ax.set_xticks(powers_of_2)
ax.set_xticklabels([str(x) for x in powers_of_2])
ax.set_xlim(0.8, 80)
ax.set_ylim(bottom=0)

ax.set_xlabel('Frame Skip', fontsize=12)
ax.set_ylabel('Runtime (minutes)', fontsize=12)
ax.set_title('Haul Scan Runtime: Decode Method × Prefetch (Mac Studio M4 Max)', fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
ax.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('mac_studio_decode_comparison.png', dpi=150, bbox_inches='tight')
print("图1 已保存: mac_studio_decode_comparison.png")


# === 图2a: Scaling - Decode All ===
fig, ax = plt.subplots(figsize=(10, 6))

baseline_old_pf = df_old[df_old['frame_skip'] == 1]['runtime_prefetch'].values[0]
baseline_old_no = df_old[df_old['frame_skip'] == 1]['runtime_no_prefetch'].values[0]
df_old['runtime_pct_prefetch'] = (df_old['runtime_prefetch'] / baseline_old_pf) * 100
df_old['runtime_pct_no_prefetch'] = (df_old['runtime_no_prefetch'] / baseline_old_no) * 100
ideal_pct = (1 / df_old['frame_skip']) * 100

powers_of_2_y = [100, 50, 25, 12.5, 6.25, 3.125]

ax.plot(df_old['frame_skip'], ideal_pct, '--', color='#2ecc71', linewidth=2, alpha=0.7, label='Ideal (runtime ∝ 1/skip)')
ax.plot(df_old['frame_skip'], df_old['runtime_pct_no_prefetch'], 'o-', color='#e74c3c', linewidth=2, markersize=4, label='No Prefetch')
ax.plot(df_old['frame_skip'], df_old['runtime_pct_prefetch'], 's-', color='#3498db', linewidth=2, markersize=4, label='With Prefetch')

ax.set_xscale('log', base=2)
ax.set_yscale('log', base=2)
ax.set_xticks(powers_of_2)
ax.set_xticklabels([str(x) for x in powers_of_2])
ax.set_yticks(powers_of_2_y)
ax.set_yticklabels([f'{y:.2f}%' if y < 10 else f'{int(y)}%' for y in powers_of_2_y])
ax.set_xlim(0.8, 80)
ax.set_ylim(2, 120)

ax.set_xlabel('Frame Skip', fontsize=12)
ax.set_ylabel('Runtime (% of skip=1)', fontsize=12)
ax.set_title('Runtime Scaling: Decode All (Mac Studio M4 Max)', fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
ax.grid(True, alpha=0.3, linestyle='--', which='both')

plt.tight_layout()
plt.savefig('mac_studio_scaling_old_decode.png', dpi=150, bbox_inches='tight')
print("图2a 已保存: mac_studio_scaling_old_decode.png")


# === 图2b: Scaling - Decode On Demand ===
fig, ax = plt.subplots(figsize=(10, 6))

baseline_eff_pf = df_eff[df_eff['frame_skip'] == 1]['runtime_prefetch'].values[0]
baseline_eff_no = df_eff[df_eff['frame_skip'] == 1]['runtime_no_prefetch'].values[0]
df_eff['runtime_pct_prefetch'] = (df_eff['runtime_prefetch'] / baseline_eff_pf) * 100
df_eff['runtime_pct_no_prefetch'] = (df_eff['runtime_no_prefetch'] / baseline_eff_no) * 100

ax.plot(df_eff['frame_skip'], ideal_pct, '--', color='#2ecc71', linewidth=2, alpha=0.7, label='Ideal (runtime ∝ 1/skip)')
ax.plot(df_eff['frame_skip'], df_eff['runtime_pct_no_prefetch'], 'o-', color='#e74c3c', linewidth=2, markersize=4, label='No Prefetch')
ax.plot(df_eff['frame_skip'], df_eff['runtime_pct_prefetch'], 's-', color='#3498db', linewidth=2, markersize=4, label='With Prefetch')

ax.set_xscale('log', base=2)
ax.set_yscale('log', base=2)
ax.set_xticks(powers_of_2)
ax.set_xticklabels([str(x) for x in powers_of_2])
ax.set_yticks(powers_of_2_y)
ax.set_yticklabels([f'{y:.2f}%' if y < 10 else f'{int(y)}%' for y in powers_of_2_y])
ax.set_xlim(0.8, 80)
ax.set_ylim(2, 120)

ax.set_xlabel('Frame Skip', fontsize=12)
ax.set_ylabel('Runtime (% of skip=1)', fontsize=12)
ax.set_title('Runtime Scaling: Decode On Demand (Mac Studio M4 Max)', fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
ax.grid(True, alpha=0.3, linestyle='--', which='both')

plt.tight_layout()
plt.savefig('mac_studio_scaling_efficient_decode.png', dpi=150, bbox_inches='tight')
print("图2b 已保存: mac_studio_scaling_efficient_decode.png")

# === 图2c: Scaling Combined (side-by-side) ===
fig, axes = plt.subplots(figsize=(18, 6), ncols=2)
img_left = plt.imread('mac_studio_scaling_old_decode.png')
img_right = plt.imread('mac_studio_scaling_efficient_decode.png')
axes[0].imshow(img_left)
axes[0].axis('off')
axes[0].set_title('', fontsize=13, fontweight='bold')
axes[1].imshow(img_right)
axes[1].axis('off')
axes[1].set_title('', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('mac_studio_scaling_combined.png', dpi=150, bbox_inches='tight')
print("图2c 已保存: mac_studio_scaling_combined.png")


# === 图3a: Time Savings - Decode All ===
fig, ax1 = plt.subplots(figsize=(10, 6))

df_old['time_saved_sec'] = df_old['runtime_no_prefetch'] - df_old['runtime_prefetch']
df_old['time_saved_pct'] = (df_old['time_saved_sec'] / df_old['runtime_no_prefetch']) * 100

color1 = '#3498db'
ax1.bar(df_old['frame_skip'], df_old['time_saved_sec'], color=color1, alpha=0.6, label='Time Saved (seconds)')
ax1.set_xlabel('Frame Skip', fontsize=12)
ax1.set_ylabel('Time Saved (seconds)', fontsize=12, color=color1)
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_ylim(0, 500)

avg_saved = df_old['time_saved_sec'].mean()
ax1.axhline(y=avg_saved, color=color1, linestyle='--', linewidth=2, alpha=0.8)
ax1.text(65, avg_saved + 15, f'Avg: {avg_saved:.0f}s', color=color1, fontsize=10, ha='right')

ax2 = ax1.twinx()
color2 = '#e67e22'
ax2.plot(df_old['frame_skip'], df_old['time_saved_pct'], 'o-', color=color2, linewidth=2, markersize=4, label='Time Saved (%)')
ax2.set_ylabel('Time Saved (%)', fontsize=12, color=color2)
ax2.tick_params(axis='y', labelcolor=color2)
ax2.set_ylim(0, 55)

peak_idx = df_old['time_saved_pct'].idxmax()
peak_skip = df_old.loc[peak_idx, 'frame_skip']
peak_pct = df_old.loc[peak_idx, 'time_saved_pct']
ax2.annotate(f'Peak: {peak_pct:.1f}%\n(skip={peak_skip:.0f})',
             xy=(peak_skip, peak_pct), xytext=(peak_skip + 8, peak_pct + 3),
             fontsize=10, color=color2,
             arrowprops=dict(arrowstyle='->', color=color2, lw=1.5))

ax1.set_xticks([1, 8, 16, 24, 32, 40, 48, 56, 64])
ax1.set_xlim(0, 65)

plt.title('Prefetch Time Savings: Decode All (Mac Studio M4 Max)', fontsize=14, fontweight='bold')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=10)

plt.tight_layout()
plt.savefig('mac_studio_time_savings_old_decode.png', dpi=150, bbox_inches='tight')
print("图3a 已保存: mac_studio_time_savings_old_decode.png")


# === 图3b: Time Savings - Decode On Demand ===
fig, ax1 = plt.subplots(figsize=(10, 6))

df_eff['time_saved_sec'] = df_eff['runtime_no_prefetch'] - df_eff['runtime_prefetch']
df_eff['time_saved_pct'] = (df_eff['time_saved_sec'] / df_eff['runtime_no_prefetch']) * 100

color1 = '#3498db'
ax1.bar(df_eff['frame_skip'], df_eff['time_saved_sec'], color=color1, alpha=0.6, label='Time Saved (seconds)')
ax1.set_xlabel('Frame Skip', fontsize=12)
ax1.set_ylabel('Time Saved (seconds)', fontsize=12, color=color1)
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_ylim(0, 500)

avg_saved = df_eff['time_saved_sec'].mean()
ax1.axhline(y=avg_saved, color=color1, linestyle='--', linewidth=2, alpha=0.8)
ax1.text(65, avg_saved + 15, f'Avg: {avg_saved:.0f}s', color=color1, fontsize=10, ha='right')

ax2 = ax1.twinx()
color2 = '#e67e22'
ax2.plot(df_eff['frame_skip'], df_eff['time_saved_pct'], 'o-', color=color2, linewidth=2, markersize=4, label='Time Saved (%)')
ax2.set_ylabel('Time Saved (%)', fontsize=12, color=color2)
ax2.tick_params(axis='y', labelcolor=color2)
ax2.set_ylim(0, 55)

peak_idx = df_eff['time_saved_pct'].idxmax()
peak_skip = df_eff.loc[peak_idx, 'frame_skip']
peak_pct = df_eff.loc[peak_idx, 'time_saved_pct']
ax2.annotate(f'Peak: {peak_pct:.1f}%\n(skip={peak_skip:.0f})',
             xy=(peak_skip, peak_pct), xytext=(peak_skip + 8, peak_pct + 3),
             fontsize=10, color=color2,
             arrowprops=dict(arrowstyle='->', color=color2, lw=1.5))

ax1.set_xticks([1, 8, 16, 24, 32, 40, 48, 56, 64])
ax1.set_xlim(0, 65)

plt.title('Prefetch Time Savings: Decode On Demand (Mac Studio M4 Max)', fontsize=14, fontweight='bold')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=10)

plt.tight_layout()
plt.savefig('mac_studio_time_savings_efficient_decode.png', dpi=150, bbox_inches='tight')
print("图3b 已保存: mac_studio_time_savings_efficient_decode.png")


# 统计
print("\n=== 统计 ===")
print(f"Decode All:       {total_old_no:.1f}h -> {total_old_pf:.1f}h (prefetch saves {(1-total_old_pf/total_old_no)*100:.1f}%)")
print(f"Decode On Demand: {total_eff_no:.1f}h -> {total_eff_pf:.1f}h (prefetch saves {(1-total_eff_pf/total_eff_no)*100:.1f}%)")
print(f"Overall: Decode All no-prefetch {total_old_no:.1f}h -> Decode On Demand prefetch {total_eff_pf:.1f}h (saves {(1-total_eff_pf/total_old_no)*100:.1f}%)")
