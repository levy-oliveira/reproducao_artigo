from dataset import(
    load_dataset,
    validate_dataset
) 
from pathlib import Path
from rapidfuzz import fuzz

ROOT_DIR = Path(__file__).resolve().parent.parent

FILE_NAME = "predictions_returnTag_10_SR"

OUTPUT_DIR = ROOT_DIR / "outputs"
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
    if item['match']:       
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


output_file = (ROOT_DIR / f"src/metricas/qwen/{FILE_NAME}.txt")
count = 0
with open(output_file, 'a') as file:
    # for ctv in correct_to_validate:
    #     count +=1
    #     file.write(f"\n\nItem: {count}\nId:{ctv[2]}\nGround truth: {ctv[0]}\nPrediction: {ctv[1]}")
    
    # file.write(f"\n\nTotal para validar: {len(correct_to_validate)}\n\n")
   file.write(f"Total: {total}\n\nPredicoes corretas exatas: {correct_prediction}\n" +
    f"Acuracia das predicoes exatas: {(correct_prediction / total):.2f}\n")
   file.write(f"Numero de incorretas: {len(incorrect)}\n")
   non_validate_correct_prediction -= 17
   file.write(f"Numero de semanticamente corretas: {non_validate_correct_prediction}\n" + 
               f"Acuracia: {(non_validate_correct_prediction / total):.2f}\n")
