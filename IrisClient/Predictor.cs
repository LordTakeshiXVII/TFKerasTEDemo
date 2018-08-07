using System.IO;
using TensorFlow;

namespace IrisClient
{
    static class Predictor
    {
        public static IrisPrediction Predict(IrisFeatures features)
        {
            var data = features.ToArray();
            var tensor = TFTensor.FromBuffer(new TFShape(1, data.Length), data, 0, data.Length);

            using (var graph = new TFGraph())
            {
                graph.Import(File.ReadAllBytes("keras_frozen.pb"));
                var session = new TFSession(graph);
                var runner = session.GetRunner();
                runner.AddInput(graph["input_layer"][0], tensor);
                runner.Fetch(graph["output_layer/Softmax"][0]);
                var output = runner.Run();
                TFTensor result = output[0];
                float[] p = ((float[][])result.GetValue(true))[0];
                return IrisPrediction.FromArray(p);    
            }
        }
    }
}
