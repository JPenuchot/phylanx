// Copyright (c) 2019 Shahrzad Shirzad
// Copyright (c) 2019 Hartmut Kaiser
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <phylanx/config.hpp>
#include <phylanx/execution_tree/primitives/node_data_helpers.hpp>
#include <phylanx/ir/node_data.hpp>
#include <phylanx/plugins/keras_support/relu.hpp>
#include <phylanx/util/detail/numeric_limits_min.hpp>
#include <phylanx/util/matrix_iterators.hpp>

#include <hpx/include/lcos.hpp>
#include <hpx/include/naming.hpp>
#include <hpx/include/util.hpp>
#include <hpx/throw_exception.hpp>
#include <hpx/util/iterator_facade.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <blaze/Math.h>
#if defined(PHYLANX_HAVE_BLAZE_TENSOR)
#include <blaze_tensor/Math.h>
#endif
///////////////////////////////////////////////////////////////////////////////
namespace phylanx { namespace execution_tree { namespace primitives
{
    ///////////////////////////////////////////////////////////////////////////
    match_pattern_type const relu::match_data = {match_pattern_type{"relu",
        std::vector<std::string>{"relu(_1)", "relu(_1, _2)", "relu(_1, _2, _3)",
            "relu(_1, _2, _3, _4)"},
        &create_relu, &create_primitive<relu>,
        R"(
            x, alpha, max_value, threshold
            Args:

                x (array_like) : array
                alpha : Slope of negative region, scalar default is 0.0
                max_value : Saturation threshold, scalar
                threshold : Threshold for thresholded activations, scalar

            Returns:

            An array with the elements of a, but where values < a_min are replaced with
            a_min, and those > a_max with a_max."
            )"}};

    ///////////////////////////////////////////////////////////////////////////
    relu::relu(primitive_arguments_type&& operands, std::string const& name,
        std::string const& codename)
      : primitive_component_base(std::move(operands), name, codename)
    {
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    primitive_argument_type relu::relu0d(ir::node_data<T>&& arg, double& alpha,
        T& max_value, double& threshold) const
    {
        auto a = arg.scalar();
        if (a < threshold)
            a = alpha * (a - threshold);
        else
            a = (blaze::max)(T(0), (blaze::min)(a, max_value));
        return primitive_argument_type{std::move(a)};
    }

    template <typename T>
    primitive_argument_type relu::relu1d(ir::node_data<T>&& arg, double& alpha,
        T& max_value, double& threshold) const
    {
        auto v = arg.vector();
        std::transform(v.data(), v.data() + v.size(), v.data(), [&](T a) -> T {
            if (a < threshold)
                a = alpha * (a - threshold);
            else
                a = (blaze::max)(T(0), (blaze::min)(a, max_value));
            return a;
        });
        return primitive_argument_type{std::move(arg)};
    }

    template <typename T>
    primitive_argument_type relu::relu2d(ir::node_data<T>&& arg, double& alpha,
        T& max_value, double& threshold) const
    {
        auto m = arg.matrix();

        using phylanx::util::matrix_row_iterator;
        const matrix_row_iterator<decltype(m)> m_begin(m);
        const matrix_row_iterator<decltype(m)> m_end(m, m.rows());

        for (auto it = m_begin; it != m_end; ++it)
        {
            std::transform(it->begin(), it->end(), it->begin(), [&](T a) -> T {
                if (a < threshold)
                    a = alpha * (a - threshold);
                else
                    a = (blaze::max)(T(0), (blaze::min)(a, max_value));
                return a;
            });
        }
        return primitive_argument_type{std::move(arg)};
    }

#if defined(PHYLANX_HAVE_BLAZE_TENSOR)
    template <typename T>
    primitive_argument_type relu::relu3d(ir::node_data<T>&& arg, double& alpha,
        T& max_value, double& threshold) const
    {
        auto t = arg.tensor();
        using phylanx::util::matrix_row_iterator;
        for (std::size_t i = 0; i != t.pages(); ++i)
        {
            auto slice = blaze::pageslice(t, i);
            matrix_row_iterator<decltype(slice)> t_begin(slice);
            matrix_row_iterator<decltype(slice)> t_end(slice, slice.rows());

            for (auto it = t_begin; it != t_end; ++it)
                std::transform(
                    it->begin(), it->end(), it->begin(), [&](T a) -> T {
                        if (a < threshold)
                            a = alpha * (a - threshold);
                        else
                            a = (blaze::max)(T(0), (blaze::min)(a, max_value));
                        return a;
                    });
        }

        return primitive_argument_type{std::move(arg)};
    }
#endif

    template <typename T>
    primitive_argument_type relu::relu_helper(ir::node_data<T>&& arg,
        double& alpha, T& max_value, double& threshold) const
    {
        switch (arg.num_dimensions())
        {
        case 0:
            return relu0d(
                std::move(arg), alpha, max_value, threshold);
        case 1:
            return relu1d(
                std::move(arg), alpha, max_value, threshold);
        case 2:
            return relu2d(
                std::move(arg), alpha, max_value, threshold);
#if defined(PHYLANX_HAVE_BLAZE_TENSOR)
        case 3:
            return relu3d(
                std::move(arg), alpha, max_value, threshold);
#endif
        default:
            break;
        }
        HPX_THROW_EXCEPTION(hpx::bad_parameter,
            "relu::relu_helper",
            generate_error_message("invalid dimension"));
        auto this_ = this->shared_from_this();
    }

    ///////////////////////////////////////////////////////////////////////////
    hpx::future<primitive_argument_type> relu::eval(
        primitive_arguments_type const& operands,
        primitive_arguments_type const& args, eval_context ctx) const
    {
        if (!valid(operands[0]))
            HPX_THROW_EXCEPTION(hpx::bad_parameter,
                "relu::eval",
                generate_error_message("the relu primitive requires that the "
                                       "first argument is valid"));
        if (operands.empty() || operands.size() > 4)
        {
            HPX_THROW_EXCEPTION(hpx::bad_parameter,
                "relu::eval",
                generate_error_message(
                    "the relu primitive requires at most four operands"));
        }

        auto this_ = this->shared_from_this();
        return hpx::dataflow(hpx::launch::sync,
            hpx::util::unwrapping([this_ = std::move(this_)](
                                      primitive_arguments_type&& args)
                                      -> primitive_argument_type {
                double alpha = 0.;
                double threshold = 0.;

                if (args.size() > 1)
                {
                    auto a = extract_numeric_value(
                        std::move(args[1]), this_->name_, this_->codename_);
                    alpha = a.scalar();
                }

                if (args.size() > 3)
                {
                    auto t = extract_numeric_value(
                        std::move(args[3]), this_->name_, this_->codename_);
                    threshold = t.scalar();
                }

                node_data_type t = extract_common_type(args[0]);

                switch (t)
                {
                case node_data_type_int64:
                {
                    std::int64_t max_value;
                    if (args.size() == 2 || !valid(args[2]))
                        max_value = (std::numeric_limits<std::int64_t>::max)();
                    else
                        max_value = extract_scalar_integer_value(
                            std::move(args[2]), this_->name_, this_->codename_);
                    return this_->relu_helper<std::int64_t>(
                        extract_integer_value_strict(
                            std::move(args[0]), this_->name_, this_->codename_),
                        alpha, max_value, threshold);
                }
                case node_data_type_bool:
                {
                    std::uint8_t max_value;
                    if (args.size() == 2 || !valid(args[2]))
                        max_value = (std::numeric_limits<std::uint8_t>::max)();
                    else
                    {
                        auto m = extract_boolean_value(
                            std::move(args[2]), this_->name_, this_->codename_);
                        max_value = m.scalar();
                    }
                    return this_->relu_helper<std::uint8_t>(
                        extract_boolean_value_strict(
                            std::move(args[0]), this_->name_, this_->codename_),
                        alpha, max_value, threshold);
                }
                case node_data_type_unknown:
                    HPX_FALLTHROUGH;
                case node_data_type_double:
                {
                    double max_value;
                    if (args.size() == 2 || !valid(args[2]))
                        max_value = (std::numeric_limits<double>::max)();
                    else
                    {
                        auto m = extract_numeric_value(
                            std::move(args[2]), this_->name_, this_->codename_);
                        max_value = m.scalar();
                    }

                    return this_->relu_helper<double>(
                        extract_numeric_value_strict(
                            std::move(args[0]), this_->name_, this_->codename_),
                        alpha, max_value, threshold);
                }
                default:
                    break;
                }
                HPX_THROW_EXCEPTION(hpx::bad_parameter,
                    "relu::eval",
                    this_->generate_error_message(
                        "the relu primitive requires for all arguments to "
                        "be numeric data types"));
            }),
            detail::map_operands(operands, functional::value_operand{}, args,
                name_, codename_, std::move(ctx)));

        HPX_THROW_EXCEPTION(hpx::bad_parameter,
            "relu::eval",
            this_->generate_error_message("unsupported mode requested"));
    }
}}}
