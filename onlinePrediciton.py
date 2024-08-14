# Online Python compiler (interpreter) to run Python online.
# Write Python 3 code in this online editor and run it.
# TODO(developer): Uncomment and set the following variables







project_id = 'smart-chair-307219'
#compute_region = 'Global'
compute_region = "us-central1"
#GOOGLE_APPLICATION_CREDENTIALS = "C:\Users\Sean\Downloads\smart-chair-307219-34ce5caf904d.json"
#export GOOGLE_APPLICATION_CREDENTIALS="C:\Users\Sean\Desktop\Senior Design\smart-chair-307219-34ce5caf904d.json"
#for command line
#set GOOGLE_APPLICATION_CREDENTIALS = "C:\Users\Sean\Downloads\smart-chair-307219-34ce5caf904d.json"
model_display_name = 'BigData_20210418125437'
inputs = {'FSR0': 3, 'FSR1': 3, 'FSR2': 3, 'FSR3': 3, 'FSR4': 3, 'FSR5': 3, 'FSR6': 3, 'FSR7': 3}

from google.cloud import automl_v1beta1 as automl
from google.oauth2 import service_account
client = automl_v1beta1.TablesClient(credentials=service_account.Credentials.from_service_account_file("C:\Users\Sean\Desktop\Senior Design\smart-chair-307219-34ce5caf904d.json"), project='my-project', region='us-central1')




client = automl.TablesClient(project=project_id, region=compute_region)

if feature_importance:
    response = client.predict(
        model_display_name=model_display_name,
        inputs=inputs,
        feature_importance=True,
    )
else:
    response = client.predict(
        model_display_name=model_display_name, inputs=inputs
    )

print("Prediction results:")
for result in response.payload:
    print(
        "Predicted class name: {}".format(result.tables.value)
    )
    print("Predicted class score: {}".format(result.tables.score))

    if feature_importance:
        # get features of top importance
        feat_list = [
            (column.feature_importance, column.column_display_name)
            for column in result.tables.tables_model_column_info
        ]
        feat_list.sort(reverse=True)
        if len(feat_list) < 10:
            feat_to_show = len(feat_list)
        else:
            feat_to_show = 10

        print("Features of top importance:")
        for feat in feat_list[:feat_to_show]:
            print(feat)