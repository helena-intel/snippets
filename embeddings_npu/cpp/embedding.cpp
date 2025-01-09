#include <openvino/openvino.hpp>
#include <iostream>
#include <filesystem>

int main(int argc, char* argv[]) {

   if (3 != argc) {
       throw std::runtime_error(std::string{"Usage: "} + argv[0] + " <MODEL_DIR> <DEVICE>");
   }

   std::string dirname = argv[1];
   std::string device = argv[2];
   std::filesystem::path dir_path(dirname);
   std::filesystem::path model_xml = dir_path / "openvino_model.xml";
   std::filesystem::path tokenizer_xml = dir_path / "openvino_tokenizer.xml";

   ov::Core core;
   // Windows: "openvino_tokenizers.dll", Linux: "libopenvino_tokenizers.so" MacOS: "libopenvino_tokenizers.dylib"
   #if defined(_WIN32) || defined(_WIN64)
       core.add_extension("openvino_tokenizers.dll");
   #elif defined(__linux__)
       core.add_extension("libopenvino_tokenizers.so");
   #endif

   std::vector<std::string> prompts = {"Hello world!", "OpenVINO is great"};
   std::size_t batch_size = prompts.size();

   ov::InferRequest tokenizer_request = core.compile_model(tokenizer_xml, "CPU").create_infer_request();
   tokenizer_request.set_input_tensor(ov::Tensor{ov::element::string, ov::Shape{batch_size}, &prompts[0]});
   tokenizer_request.infer();
   ov::Tensor input_ids = tokenizer_request.get_tensor("input_ids");
   ov::Tensor attention_mask = tokenizer_request.get_tensor("attention_mask");

   auto ov_model = core.read_model(model_xml.string());
   ov::set_batch(ov_model, batch_size);
   ov::InferRequest infer_request = core.compile_model(ov_model, device).create_infer_request();
   infer_request.set_tensor("input_ids", input_ids);
   infer_request.set_tensor("attention_mask", attention_mask);

   // Add token_type_ids if needed.
   try {
       ov::Tensor token_type_ids = tokenizer_request.get_tensor("token_type_ids");
	   infer_request.set_tensor("token_type_ids", token_type_ids);
   } catch (const std::exception& e) {}

   infer_request.infer();

   auto output = infer_request.get_tensor("last_hidden_state");
   const float *output_buffer = output.data<const float>();

   for (int i = 0; i < batch_size; ++i) {
      for (size_t j = 0; j < 10; j++) {
         std::cout << std::fixed << std::setprecision(4) << output_buffer[i * output.get_size() / batch_size + j] << " ";
      }
      std::cout << std::endl;
   }
   return 0;
}
