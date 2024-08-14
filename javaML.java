import com.google.cloud.automl.v1beta1.AnnotationPayload;
import com.google.cloud.automl.v1beta1.ExamplePayload;
import com.google.cloud.automl.v1beta1.ModelName;
import com.google.cloud.automl.v1beta1.PredictRequest;
import com.google.cloud.automl.v1beta1.PredictResponse;
import com.google.cloud.automl.v1beta1.PredictionServiceClient;
import com.google.cloud.automl.v1beta1.Row;
import com.google.cloud.automl.v1beta1.TablesAnnotation;
import com.google.protobuf.Value;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

class TablesPredict {

  public static void main(String[] args) throws IOException {
    // TODO(developer): Replace these variables before running the sample.
    String projectId = "smart-chair-307219";
    String modelId = "TBL4460245883887288320";
    // Values should match the input expected by your model.
    List<Value> values = new ArrayList<>();
    // values.add(Value.newBuilder().setBoolValue(true).build());
    // values.add(Value.newBuilder().setNumberValue(10).build());
    // values.add(Value.newBuilder().setStringValue("YOUR_STRING").build());
    predict(projectId, modelId, values);
  }

  static void predict(String projectId, String modelId, List<Value> values) throws IOException {
    // Initialize client that will be used to send requests. This client only needs to be created
    // once, and can be reused for multiple requests. After completing all of your requests, call
    // the "close" method on the client to safely clean up any remaining background resources.
    try (PredictionServiceClient client = PredictionServiceClient.create()) {
      // Get the full path of the model.
      ModelName name = ModelName.of(projectId, "us-central1", modelId);
      Row row = Row.newBuilder().addAllValues(values).build();
      ExamplePayload payload = ExamplePayload.newBuilder().setRow(row).build();

      // Feature importance gives you visibility into how the features in a specific prediction
      // request informed the resulting prediction. For more info, see:
      // https://cloud.google.com/automl-tables/docs/features#local
      PredictRequest request =
          PredictRequest.newBuilder()
              .setName(name.toString())
              .setPayload(payload)
              .putParams("feature_importance", "true")
              .build();

      PredictResponse response = client.predict(request);

      System.out.println("Prediction results:");
      for (AnnotationPayload annotationPayload : response.getPayloadList()) {
        TablesAnnotation tablesAnnotation = annotationPayload.getTables();
        System.out.format(
            "Classification label: %s%n", tablesAnnotation.getValue().getStringValue());
        System.out.format("Classification score: %.3f%n", tablesAnnotation.getScore());
        // Get features of top importance
        tablesAnnotation
            .getTablesModelColumnInfoList()
            .forEach(
                info ->
                    System.out.format(
                        "\tColumn: %s - Importance: %.2f%n",
                        info.getColumnDisplayName(), info.getFeatureImportance()));
      }
    }
  }
}