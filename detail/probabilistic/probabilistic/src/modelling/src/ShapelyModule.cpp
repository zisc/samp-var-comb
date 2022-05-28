#include <string>
#include <utility>
#include <vector>
#include <log/trivial.hpp>
#include <torch/torch.h>
#include <modelling/model/ShapelyModule.hpp>

std::vector<torch::Tensor> ShapelyModule::parameters(bool recurse, bool include_fixed) const {
    return torch::nn::Module::parameters(recurse);
}

std::vector<torch::Tensor> ShapelyModule::parameters_on_paper(bool recurse) const {
    auto parameters_on_paper_dict_values = parameters_on_paper_dict.values();
    std::vector<torch::Tensor> ret; ret.reserve(parameters_on_paper_dict_values.size());
    for (const auto& p : parameters_on_paper_dict_values) {
        ret.emplace_back(p->get());
    }
    return ret;
}

torch::OrderedDict<std::string, torch::Tensor> ShapelyModule::named_parameters(bool recurse, bool include_fixed) const {
    return torch::nn::Module::named_parameters(recurse);
}

torch::OrderedDict<std::string, torch::Tensor> ShapelyModule::named_parameters_on_paper(bool recurse) const {
    torch::OrderedDict<std::string, torch::Tensor> ret;
    ret.reserve(parameters_on_paper_dict.size());
    for (const auto& p : parameters_on_paper_dict) {
        ret.insert(p.key(), p.value()->get());
    }
    if (recurse) {
        for (const auto& m : shapely_modules_dict) {
            auto prefix = m.key() + '.';
            auto without_prefix = m.value()->named_parameters_on_paper(recurse);
            ret.reserve(ret.size() + without_prefix.size());
            for (auto& p : without_prefix) {
                ret.insert(prefix + p.key(), std::move(p.value()));
            }
        }
    }
    return ret;
}

void ShapelyModule::parameter_dump(bool recursive) const {
    for (const auto& p : parameters_on_paper_dict) {
        PROBABILISTIC_LOG_TRIVIAL_DEBUG << p.value()->get() << " = " << p.key() << '\n';
    }
    if (recursive) {
        for (const auto& m : shapely_modules_dict) {
            m.value()->parameter_dump(recursive);
        }
    }

    auto raw_parameters = named_parameters();
    for (const auto& p : raw_parameters) {
        PROBABILISTIC_LOG_TRIVIAL_DEBUG << p.value() << " = " << p.key() << '\n';
    }

    auto raw_buffers = named_buffers();
    for (const auto& b : raw_buffers) {
        PROBABILISTIC_LOG_TRIVIAL_DEBUG << b.value() << " = " << b.key() << '\n';
    }
}

void ShapelyModule::set_parameters(const torch::OrderedDict<std::string, torch::Tensor>& new_parameters) {
    auto this_parameters = named_parameters(/*recurse =*/ true);
    for (const auto& new_item : new_parameters ) {
        auto *to_override = this_parameters.find(new_item.key());
        if (to_override) {
            to_override->set_data(new_item.value());
        }
    }
}

