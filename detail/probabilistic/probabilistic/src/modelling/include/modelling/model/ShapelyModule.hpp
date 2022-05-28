#ifndef PROBABILISTIC_SHAPELY_MODULE_HPP_GUARD
#define PROBABILISTIC_SHAPELY_MODULE_HPP_GUARD

#include <memory>
#include <string>
#include <type_traits>
#include <vector>
#include <torch/torch.h>
#include <libtorch_support/Parameterisation.hpp>
#include <libtorch_support/Buffers.hpp>

inline std::string shapely_parameter_raw_name(const std::string& name) {
    return name + "_shapely";
}

inline std::string shapely_parameter_raw_name(const Parameterisation& p) {
    return shapely_parameter_raw_name(p.name());
}

class ShapelyModule : public virtual torch::nn::Module {
    template<class Derived>
    friend class ShapelyCloneable;

    public:
        // Deleting the move assignment operator prevents repeatedly moving from a virtual base ShapelyModule. See
        // https://stackoverflow.com/questions/46599222/why-does-gcc-warn-about-calling-a-non-trivial-move-assignment-operator-with-std
        ShapelyModule& operator=(ShapelyModule&&) = delete;

        // Since the move assignment operator is deleted, we must manually declare the default constructor, copy constructor, and
        // copy assignment operator.
        ShapelyModule() = default;
        ShapelyModule(const ShapelyModule&) = default;
        ShapelyModule& operator=(const ShapelyModule&) = default;

        virtual std::vector<torch::Tensor> parameters(bool recurse = true, bool include_fixed = true) const;

        virtual std::vector<torch::Tensor> parameters_on_paper(bool recurse = true) const;

        virtual torch::OrderedDict<std::string, torch::Tensor> named_parameters(bool recurse = true, bool include_fixed = true) const;

        virtual torch::OrderedDict<std::string, torch::Tensor> named_parameters_on_paper(bool recurse = true) const;

        void parameter_dump(bool recursive = true) const;

        void set_parameters(const torch::OrderedDict<std::string, torch::Tensor>& new_parameters);

    protected:
        template<class ModuleType>
        std::shared_ptr<ModuleType> register_module(std::string name, std::shared_ptr<ModuleType> module) {
            static_assert(std::is_base_of<ShapelyModule, ModuleType>::value, "Only ShapelyModules can be registered by a ShapelyModule.");
            shapely_modules_dict.insert(name, module);
            return torch::nn::Module::register_module(std::move(name), std::move(module));
        }

        template<class ModuleType>
        std::shared_ptr<ModuleType> register_module(std::shared_ptr<ModuleType> module) {
            auto name = module->name();
            return register_module(std::move(name), std::move(module));
        }

        // You cannot register a null parameter for later overriding by deserialisation, because libtorch
        // will not honour the requires_grad = true parameter. Instead, it is the responsibility of the
        // caller to ensure that the module is created with the right parameters for the serialised
        // object.
        /*
        template<class P>
        std::shared_ptr<P> register_null_shapely_parameter(std::string name, bool requires_grad = true) {
            // auto p = std::make_shared<P>(...) throws an exception.
            return std::shared_ptr<P>(new P(
                register_buffer(name + "_enabled", torch::Tensor()),
                name,
                register_parameter(name + "_shapely", torch::Tensor(), requires_grad),
                register_buffer(name + "_lower_bound", torch::Tensor()),
                register_buffer(name + "_upper_bound", torch::Tensor()),
                register_buffer(name + "_scaling", torch::Tensor()),
                register_buffer(name + "_barrier_scaling", torch::Tensor())
            ));
        }
        */

        template<class P>
        std::shared_ptr<P> register_shapely_parameter(std::string name, ShapelyParameter shapely_parameter = {}, bool requires_grad = true, bool as_buffer = false) {
            if (shapely_parameter.enable) {
                requires_grad = requires_grad && shapely_parameter.parameter.numel();
                // auto p = std::make_shared<P>(...) throws an exception.
                auto p = std::shared_ptr<P>(new P(
                    register_buffer(name + "_enabled", torch::full({1}, shapely_parameter.enable, torch::kBool)),
                    name,
                    as_buffer ? register_buffer(shapely_parameter_raw_name(name), std::move(shapely_parameter.parameter))
                              : register_parameter(shapely_parameter_raw_name(name), std::move(shapely_parameter.parameter), requires_grad),
                    register_buffer(name + "_lower_bound", torch::full({1}, shapely_parameter.lower_bound, torch::kDouble)),
                    register_buffer(name + "_upper_bound", torch::full({1}, shapely_parameter.upper_bound, torch::kDouble)),
                    register_buffer(name + "_scaling", torch::full({1}, shapely_parameter.parameter_scaling, torch::kDouble)),
                    register_buffer(name + "_barrier_scaling", torch::full({1}, shapely_parameter.barrier_scaling, torch::kDouble))
                ));
                parameters_on_paper_dict.insert(std::move(name), p);
                return p;
            } else {
                return std::shared_ptr<P>(new P(
                    torch::full({1}, shapely_parameter.enable, torch::kBool),
                    std::move(name),
                    std::move(shapely_parameter.parameter),
                    torch::full({1}, shapely_parameter.lower_bound, torch::kDouble),
                    torch::full({1}, shapely_parameter.upper_bound, torch::kDouble),
                    torch::full({1}, shapely_parameter.parameter_scaling, torch::kDouble),
                    torch::full({1}, shapely_parameter.barrier_scaling, torch::kDouble)
                ));
            }
        }

        template<class P>
        std::shared_ptr<P> register_shapely_parameter(std::string name, const NamedShapelyParameters& shapely_parameters, bool requires_grad = true, bool as_buffer = false) {
            return register_shapely_parameter<P>(std::move(name), shapely_parameters.parameters[name], requires_grad, as_buffer);
        }

        template<class P>
        std::shared_ptr<P> register_next_shapely_parameter(NamedShapelyParameters& shapely_parameters, bool requires_grad = true, bool as_buffer = false) {
            auto this_parameter_idx = shapely_parameters.idx++; // Increment for when this function is called next time.
            auto item = shapely_parameters.parameters[this_parameter_idx];
            auto name = item.key();
            auto shapely_parameter = item.value();
            return register_shapely_parameter<P>(std::move(name), std::move(shapely_parameter), requires_grad, as_buffer);
        }

        torch::Tensor& register_next_buffer(std::string name, Buffers& buffers) {
            auto this_buffer_idx = buffers.idx++;  // Increment for when this function is called next time.
            return register_buffer(std::move(name), buffers.buffers.at(this_buffer_idx));
        }

        void observations_by_parameter_recursive_update(
            torch::OrderedDict<std::string, std::vector<torch::indexing::TensorIndex>>& to_update,
            torch::OrderedDict<std::string, std::vector<torch::indexing::TensorIndex>>&& the_update,
            const std::string& nested_module_name
        ) const {
            if (!the_update.is_empty()) {
                auto nested_module_name_dot = nested_module_name + '.';
                to_update.reserve(to_update.size() + the_update.size());
                for (auto& item : the_update) {
                    to_update.insert(nested_module_name_dot + item.key(), std::move(item.value()));
                }
            }
        }

        void observations_by_parameter_recursive_update(
            torch::OrderedDict<std::string, std::vector<std::vector<torch::indexing::TensorIndex>>>& to_update,
            torch::OrderedDict<std::string, std::vector<std::vector<torch::indexing::TensorIndex>>>&& the_update,
            const std::string& nested_module_name
        ) const {
            if (!the_update.is_empty()) {
                auto nested_module_name_dot = nested_module_name + '.';
                to_update.reserve(to_update.size() + the_update.size());
                for (auto& item : the_update) {
                    to_update.insert(nested_module_name_dot + item.key(), std::move(item.value()));
                }
            }
        }

        template<class K1, class K2, class V>
        void observations_by_parameter_recursive_update(
            torch::OrderedDict<K1, torch::OrderedDict<K2, V>>& to_update,
            torch::OrderedDict<K1, torch::OrderedDict<K2, V>>&& the_update,
            const std::string& nested_module_name
        ) const {
            to_update.reserve(to_update.size() + the_update.size());
            for (auto& item : the_update) {
                const auto& key = item.key();
                auto *to_update_value = to_update.find(key);
                if (to_update_value) {
                    observations_by_parameter_recursive_update(
                        *to_update_value,
                        std::move(item.value()),
                        nested_module_name
                    );
                } else {
                    auto& to_update_value = to_update.insert(key, torch::OrderedDict<K2, V>());
                    observations_by_parameter_recursive_update(
                        to_update_value,
                        std::move(item.value()),
                        nested_module_name
                    );
                }
            }
        }

    private:
        torch::OrderedDict<std::string, std::shared_ptr<Parameterisation>> parameters_on_paper_dict;
        torch::OrderedDict<std::string, std::shared_ptr<ShapelyModule>> shapely_modules_dict;
};

// Based on https://github.com/pytorch/pytorch/blob/4ae832e1060c72cb89de1d9693629783dbe0c9a6/torch/csrc/api/include/torch/nn/cloneable.h.
template<typename Derived>
class ShapelyCloneable : public virtual ShapelyModule, public torch::nn::Cloneable<Derived> {
    public:
        virtual void shapely_reset(void) = 0;

        void reset(void) override {
            parameters_on_paper_dict.clear();
            shapely_modules_dict.clear();
            shapely_reset();
        }
};

#endif

