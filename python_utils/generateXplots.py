import os
import re
import matplotlib.pyplot as plt
import numpy as np
import sys

def combine_data_from_folders(folders, output_folder="plots"):
    array_pattern = re.compile(r"\[([^\[\]]+)\](?!:)")  # matches content inside square brackets

    os.makedirs(output_folder, exist_ok=True)  # Create output folder if it doesn't exist

    car_id_to_data = {}

    for folder_path in folders:
        if not os.path.isdir(folder_path):
            print(f"Skipping {folder_path} (not a folder)")
            continue

        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)

            if not os.path.isfile(file_path):
                continue

            try:
                with open(file_path, 'r') as f:
                    content = f.read()

                matches = array_pattern.findall(content)

                arrays = []
                for match in matches:
                    arr = np.array([float(x) for x in match.strip().split()])
                    arrays.append(arr)

                # More descriptive title: include folder and file info
                car_id = filename.split(".")[0]
                title = f"{os.path.basename(folder_path)}_" + car_id

                if len(arrays) > 0:
                    if car_id not in car_id_to_data:
                        print(folder_path+filename)
                        car_id_to_data[car_id] = [arrays]
                    else:
                        car_id_to_data[car_id].append(arrays)

            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    return car_id_to_data


def parse_data(data, num_of_experiments):
    # data is [[predicted,expected,errors]*experiments_number]
    if len(data) != num_of_experiments:
        raise Exception("Node did not predict")
    
    if len(data[0]) != 3:
        return Exception("Node was not a cluster head in all experiments")
    
    predicted = []

    for d in data:
        predicted.append(d[0])


    errors = []
    for d in data :
        errors.append(d[2])

    # return predicted values from all experiments
    # return error values from all experiments
    # return the actual values
    return predicted, errors, data[0][1]



def main():
    parent_folder = sys.argv[1]
    # parent_folder = "./xprmnts/lr/"
    num_of_experiments = 3
    num_of_experiments = int(sys.argv[2])
    # num_of_epochs = sys.argv[3] or 50
    folders =[parent_folder+x for x in os.listdir(parent_folder)] 

    
    data = combine_data_from_folders(folders,  parent_folder)

    for d in data:
        try:
            predicted, errors, actual = parse_data(data[d],num_of_experiments)
        except Exception as e:
            print("Exception: ",e)
            continue

        average_errors = []
        for predictions in predicted:
            error = 0.0
            for i in range(len(predictions)):
                error += abs(predictions[i] - actual[i])
            average_errors.append(error/len(actual))
        # plot values
        plt.figure(figsize=(10, 4))
        for idx in range(len(predicted)):
            plt.plot(predicted[idx], label=f"Predicted temperatures "+ folders[idx].split("/")[-1]+" {:.2f}".format(average_errors[idx]) )
        plt.plot(actual, label='Actual Temperatures')
        plt.xlabel("Sample Index")
        plt.ylabel("Temperature")
        # plt.plot(predicted[1][:50], label=f"Predicted (learning_rate: 0.1)")
            
        # plt.plot(predicted[2][:50], label=f"Predicted (learning_rate: 0.01)")
        plt.title(f"{d} - Actual vs Predicted Temperatures")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(parent_folder, f"{d}_pred_vs_actual.png"))
        plt.close()

        # plot errors
        plt.figure(figsize=(10, 3))

        for idx in range(len(errors)):
            plt.plot(errors[idx], label='Training loss '+ folders[idx].split("/")[-1])

        plt.xlabel("Epochs")
        plt.ylabel("Loss(MSE)")

        plt.title(f"{d} - Training loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(parent_folder, f"{d}_training_loss.png"))
        plt.close()





if __name__ == "__main__":
    main()