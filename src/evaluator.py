from dataset import(
    load_dataset,
    validate_dataset
) 
from pathlib import Path
from rapidfuzz import fuzz
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent

FILE_NAME = "predictions_returnTag_20_SR"

OUTPUT_DIR = ROOT_DIR / "outputs" /  "llama3.1-8b" / "return"
PATH = (
    OUTPUT_DIR
    / f"{FILE_NAME}.json"
)

data = load_dataset(PATH)

total = 0
correct_prediction = 0
non_validate_correct_prediction = 0
incorrect = []
correct_to_validate = []
verify = []
for i, item in enumerate(data):

    gt = item["ground_truth"]
    pr = item["prediction"]

    total += 1    
    #if item['match']:
    if gt.__eq__(pr): 
        correct_prediction += 1
        non_validate_correct_prediction +=1
    else:
        if fuzz.ratio(gt, pr) > 50:
            non_validate_correct_prediction+=1
            correct_to_validate.append([gt, pr, item['id']])
        else:    
            incorrect.append(item['id'])

print(f"Total: {total}\n\nPredições corretas: {correct_prediction}\n" +
    f"Acurácia: {correct_prediction / total} ")

print("incorretas: ",incorrect , f", Length {len(incorrect)}")


#print("\ncorretas para validar: ",correct_to_validate , f", Length {len(correct_to_validate)}")


output_file = (ROOT_DIR / f"src/metricas/llama/{FILE_NAME}.txt")
count = 0
try:
    arg = sys.argv[1] 
except:
    print("O primeiro Argumento deve ser 'verifica' ou 'metrica'")
    exit(1) 

with open(output_file, 'a+') as file:


    if  arg is not None and arg == "verifica":
        for ctv in correct_to_validate:
            count +=1
            file.write(f"\n\nItem: {count}\nId:{ctv[2]}\nGround truth: {ctv[0]}\nPrediction: {ctv[1]}")
        
        file.write(f"\n\nTotal para validar: {len(correct_to_validate)}\n\n")
    elif arg is not None and arg == "metrica":

        file.write(f"Total: {total}\n\nPredicoes corretas exatas: {correct_prediction}\n" +
            f"Acuracia das predicoes exatas: {(correct_prediction / total):.2f}\n")
        file.write(f"Numero de incorretas: {len(incorrect)}\n")

        count = 0
        file.seek(0)

        for line in file.readlines():
            if line.__contains__("nao"):
                non_validate_correct_prediction -= 1
                count+=1
        file.write(f"\nNumero de verificacoes negativas: {count}\n")    
        file.write(f"Numero de semanticamente corretas: {non_validate_correct_prediction}\n" + 
                    f"Acuracia: {(non_validate_correct_prediction / total):.2f}\n")
    else:
        print("O primeiro Argumento deve ser 'verifica' ou 'metrica'")
        exit(1)    
