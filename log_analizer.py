import os
import re
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt


def get_time_from_progress_bar(line):
    match = re.search(r"(\d+:\d+(?::\d+)?)<", line)
    if match:
        return match.group(1)
    return None


def get_date_time(line):
    # .out: 03/27/2026 14:16:55
    m1 = re.search(r"(\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2})", line)
    if m1:
        try:
            return datetime.strptime(m1.group(1), "%m/%d/%Y %H:%M:%S")
        except:
            pass
    # .err: 2026-03-27_19:03:01
    m2 = re.search(r"(\d{4}-\d{2}-\d{2})[ _](\d{2}:\d{2}:\d{2})", line)
    if m2:
        try:
            date_str = f"{m2.group(1)} {m2.group(2)}"
            return datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
        except:
            pass
    return None


def analize_folder(logs_folder):
    data_dir = os.path.join(logs_folder, "data")
    graphs_dir = os.path.join(logs_folder, "graphs")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(graphs_dir, exist_ok=True)

    results = []
    file_out = [f for f in os.listdir(logs_folder) if f.endswith('.out')]

    for nome_out in file_out:
        base_name = nome_out.replace('.out', '')
        path_out = os.path.join(logs_folder, nome_out)
        path_err = os.path.join(logs_folder, base_name + '.err')

        params = {
            'Training Name': base_name,
            'Learning Rate': 'N/D', 'Max Steps': 'N/D', 'Grad. Acc. Steps': 'N/D',
            'Eval Batch Size': 'N/D', 'Train Batch Size': 'N/D',
            'Dropout': 0.5 if "marta" in base_name.lower() else 0.1,
            'Training Time': "N/D", 'State': 'Completed',
            '10-Mean Best Loss': None
        }

        loss_values = []
        all_timestamps = []
        real_training_time = "N/D"

        # --- 1. .OUT (parameters and Loss) ---
        with open(path_out, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                t = get_date_time(line)
                if t: all_timestamps.append(t)

                if "{'loss':" in line:
                    m = re.search(r"'loss':\s*([0-9\.]+)", line)
                    if m: loss_values.append(float(m.group(1)))

                if 'learning_rate=' in line:
                    m = re.search(r"learning_rate=([0-9\.e\-]+)", line)
                    if m: params['Learning Rate'] = m.group(1)
                if 'max_steps=' in line:
                    m = re.search(r"max_steps=([0-9]+)", line)
                    if m: params['Max Steps'] = m.group(1)
                if 'gradient_accumulation_steps=' in line:
                    m = re.search(r"gradient_accumulation_steps=([0-9]+)", line)
                    if m: params['Grad. Acc. Steps'] = m.group(1)
                if 'per_device_eval_batch_size=' in line:
                    m = re.search(r"per_device_eval_batch_size=([0-9]+)", line)
                    if m: params['Eval Batch Size'] = m.group(1)
                if 'per_device_train_batch_size=' in line:
                    m = re.search(r"per_device_train_batch_size=([0-9]+)", line)
                    if m: params['Train Batch Size'] = m.group(1)

        # --- 2. .ERR (Training Time) ---
        if os.path.exists(path_err):
            with open(path_err, 'r', encoding='utf-8', errors='ignore') as f:
                righe = f.readlines()
                for i, line in enumerate(righe):
                    if "Running Evaluation" in line:
                        for j in range(i, i - 10, -1):
                            if j >= 0:
                                t_found = get_time_from_progress_bar(righe[j])
                                if t_found:
                                    real_training_time = t_found
                                    break
                        if real_training_time != "N/D":
                            break

                            # crash
                    t_prog = get_time_from_progress_bar(line)
                    if t_prog:
                        real_training_time = t_prog

                    # Check State/Crash
                    m_steps = re.search(r"(\d+)/(\d+)\s+\[", line)
                    if m_steps:
                        current_step, total_steps = int(m_steps.group(1)), int(m_steps.group(2))
                        if current_step < total_steps:
                            params['State'] = f'Crashed ({current_step}/{total_steps})'

        # --- 3. Overall time ---
        if real_training_time != "N/D":
            params['Training Time'] = real_training_time
        elif all_timestamps:
            interval = max(all_timestamps) - min(all_timestamps)
            ore, resto = divmod(interval.total_seconds(), 3600)
            minuti, secondi = divmod(resto, 60)
            params['Training Time'] = f"{int(ore)}:{int(minuti):02}:{int(secondi):02}"

        # --- 4. LOSS AND GRAPHS ---
        if loss_values:
            params['10-Mean Best Loss'] = round(sum(sorted(loss_values)[:10]) / 10, 4)
            plt.figure(figsize=(10, 5))
            plt.plot(loss_values, color='teal')
            plt.title(f"Loss: {base_name}\n Training Time: {params['Training Time']} ({params['State']})")
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(graphs_dir, f"{base_name}.png"))
            plt.close()

        results.append(params)
        print(f"[OK] {base_name} | Training Time: {params['Training Time']} | State: {params['State']}")

    # --- 5. EXPORT ---
    if results:
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(data_dir, 'results.csv'), index=False)

        df_latex = df.copy()
        df_latex['Training Name'] = df_latex['Training Name'].str.replace('_', r'\_')
        latex_table = df_latex.to_latex(index=False, float_format="%.4f", escape=False,
                                        column_format="|l|" + "c|" * (len(df.columns) - 1))

        with open(os.path.join(data_dir, 'latex_table.txt'), 'w', encoding='utf-8') as f:
            f.write("\\begin{table}[h]\n\\centering\n\\resizebox{\\textwidth}{!}{\n")
            f.write(latex_table)
            f.write("}\n\\caption{Training Time}\n\\end{table}")


if __name__ == "__main__":
    PATH = r"C:\Users\veron\Desktop\Uni\LM\2_ANNO\AI_for_bioinformatics\graph_enc\logs"
    analize_folder(PATH)