#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <openvino/runtime/core.hpp>
#include <openssl/evp.h>

std::vector<uint8_t> aes_128_cbc_decrypt(
    std::vector<uint8_t>& cipher,
    std::vector<uint8_t>& key,
    std::vector<uint8_t>& iv) {

    EVP_CIPHER_CTX* ctx = EVP_CIPHER_CTX_new();
    EVP_DecryptInit_ex(ctx, EVP_aes_128_cbc(), nullptr, key.data(), iv.data());

    std::vector<uint8_t> plain(cipher.size() + 16);
    int len = 0, plain_size = 0;
    EVP_DecryptUpdate(ctx, plain.data(), &len, cipher.data(), static_cast<int>(cipher.size()));
    plain_size = len;
    EVP_DecryptFinal_ex(ctx, plain.data() + plain_size, &len);
    plain_size += len;
    EVP_CIPHER_CTX_free(ctx);

    plain.resize(plain_size);
    return plain;
}

std::vector<uint8_t> decrypt_file(const std::string& path,
                                  std::vector<uint8_t>& key,
                                  std::vector<uint8_t>& iv) {
    std::ifstream stream(path, std::ios::in | std::ios::binary);
    if (!stream) {
        std::cerr << "Cannot open: " << path << std::endl;
        std::exit(1);
    }
    std::vector<uint8_t> cipher((std::istreambuf_iterator<char>(stream)),
                                 std::istreambuf_iterator<char>());
    return aes_128_cbc_decrypt(cipher, key, iv);
}

std::vector<uint8_t> hex_to_bytes(const std::string& hex) {
    std::vector<uint8_t> bytes;
    for (size_t i = 0; i < hex.size(); i += 2) {
        bytes.push_back(static_cast<uint8_t>(std::stoi(hex.substr(i, 2), nullptr, 16)));
    }
    return bytes;
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <enc.xml> <key_hex> <iv_hex> [device]" << std::endl;
        return 1;
    }

    std::string xml_path = argv[1];
    std::string bin_path = xml_path.substr(0, xml_path.rfind('.')) + ".bin";
    auto key = hex_to_bytes(argv[2]);
    auto iv = hex_to_bytes(argv[3]);
    std::string device = argc > 4 ? argv[4] : "CPU";

    std::cout << "Decrypting model files..." << std::endl;
    auto model_data = decrypt_file(xml_path, key, iv);
    auto weights_data = decrypt_file(bin_path, key, iv);

    ov::Core core;
    std::string str_model(model_data.begin(), model_data.end());
    auto model = core.read_model(str_model,
        ov::Tensor(ov::element::u8, {weights_data.size()}, weights_data.data()));
    auto compiled = core.compile_model(model, device);

    std::cout << "Model compiled successfully from encrypted files." << std::endl;
    for (auto& input : model->get_parameters()) {
        std::cout << "Input: " << input->get_friendly_name()
                  << " " << input->get_shape() << std::endl;
    }
    for (auto& output : model->get_results()) {
        std::cout << "Output: " << output->get_friendly_name()
                  << " " << output->get_shape() << std::endl;
    }

    return 0;
}
