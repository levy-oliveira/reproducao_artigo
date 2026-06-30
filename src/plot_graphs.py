import os
import glob
import re
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

def parse_metric_file(filepath):
    # filepath example: src/metricas/llama/return/predictions_returnTag_10_noSR.txt
    parts = Path(filepath).parts
    
    # Depending on OS and structure, let's just find "metricas" index and go from there
    try:
        idx = parts.index("metricas")
        model = parts[idx+1]
        tag = parts[idx+2]
    except ValueError:
        return None
        
    filename = parts[-1]
    
    # Extract K and SR_Status from filename
    # e.g., predictions_returnTag_10_noSR.txt
    match = re.search(r'_(\d+)_(SR|noSR)\.txt$', filename)
    if not match:
        return None
        
    k_val = int(match.group(1))
    sr_status = match.group(2)
    
    exact_acc = None
    sem_acc = None
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
        
        # Searching for "Acuracia das predicoes exatas: 0.19"
        exact_match = re.search(r'Acuracia das predicoes exatas:\s*([0-9.]+)', content)
        if exact_match:
            exact_acc = float(exact_match.group(1)) * 100
            
        # Searching for "Acuracia: 0.55" at the bottom
        sem_match = re.search(r'Acuracia:\s*([0-9.]+)', content)
        if sem_match:
            sem_acc = float(sem_match.group(1)) * 100
            
    return {
        'Model': model,
        'Tag': tag,
        'K': k_val,
        'SR_Status': sr_status,
        'Exact_Accuracy': exact_acc,
        'Semantic_Accuracy': sem_acc
    }

def main():
    root_dir = Path(__file__).parent.parent
    metricas_dir = root_dir / "src" / "metricas"
    docs_dir = root_dir / "docs"
    
    docs_dir.mkdir(exist_ok=True)
    
    data = []
    
    # Find all txt files
    txt_files = list(metricas_dir.rglob("*.txt"))
    for file in txt_files:
        parsed = parse_metric_file(file)
        if parsed and parsed['Exact_Accuracy'] is not None and parsed['Semantic_Accuracy'] is not None:
            data.append(parsed)
            
    df = pd.DataFrame(data)
    if df.empty:
        print("Nenhum dado encontrado ou falha ao analisar.")
        return

    # To mimic the image, we want line plots.
    # We will generate plots per model and per accuracy type
    
    models = df['Model'].unique()
    acc_types = [
        ('Exact_Accuracy', 'Acuracia Exata (%)'),
        ('Semantic_Accuracy', 'Acuracia Semantica (%)')
    ]
    tags = ['param', 'return', 'throws']
    
    sns.set_theme(style="whitegrid")
    
    for model in models:
        df_model = df[df['Model'] == model]
        
        for acc_col, y_label in acc_types:
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            # fig.suptitle(f'{y_label} - {model.capitalize()}', fontsize=16)
            
            for i, tag in enumerate(tags):
                df_tag = df_model[df_model['Tag'] == tag]
                
                if df_tag.empty:
                    axes[i].set_title(f'({chr(97+i)}) {tag.capitalize()}')
                    continue
                    
                # We want SR and noSR lines
                # Usually SR is red triangle, noSR (Random/baseline) is blue triangle
                
                df_sr = df_tag[df_tag['SR_Status'] == 'SR'].sort_values('K')
                df_nosr = df_tag[df_tag['SR_Status'] == 'noSR'].sort_values('K')
                
                axes[i].plot(df_nosr['K'], df_nosr[acc_col], marker='^', markersize=8, color='#4285F4', label='noSR', linewidth=2)
                axes[i].plot(df_sr['K'], df_sr[acc_col], marker='^', markersize=8, color='#EA4335', label='SR', linewidth=2)
                
                axes[i].set_title(f'({chr(97+i)}) {tag.capitalize()}')
                axes[i].set_xlabel('K', fontweight='bold')
                axes[i].set_ylabel(y_label, fontweight='bold')
                axes[i].set_ylim(0, 100)
                axes[i].set_xlim(5, 65)
                axes[i].set_xticks([10, 20, 30, 40, 50, 60])
                
                if i == 0:
                    axes[i].legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=2, frameon=False, 
                                   handletextpad=0.5, columnspacing=1)
                    
            plt.tight_layout()
            # plt.subplots_adjust(top=0.85)
            
            # Save figure
            filename = f"grafico_{model}_{acc_col.lower()}.png"
            filepath = docs_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Gráfico salvo em: {filepath}")

if __name__ == "__main__":
    main()
