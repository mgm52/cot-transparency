from scripts.finetune_cot import FormatterOptions
from scripts.finetune_zero_shot_experiments.comparison_plot import FilterStrategy, ModelTrainMeta


FEW_SHOT = [
    # trained on few shot biases
    ModelTrainMeta(
        name="gpt-3.5-turbo",
        trained_samples=1,
        filter_strategy=FilterStrategy.correct_answer,
        train_formatters=FormatterOptions.few_shot,
    ),
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:far-ai::8C2axt31",
        trained_samples=100,
        filter_strategy=FilterStrategy.correct_answer,
        train_formatters=FormatterOptions.few_shot,
    ),
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:far-ai::8C3J2acS",
        trained_samples=1000,
        filter_strategy=FilterStrategy.correct_answer,
        train_formatters=FormatterOptions.few_shot,
    ),
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:far-ai::8C4QCudQ",
        trained_samples=10000,
        filter_strategy=FilterStrategy.correct_answer,
        train_formatters=FormatterOptions.few_shot,
    ),
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:far-ai::8C8l8MfP",
        trained_samples=20000,
        filter_strategy=FilterStrategy.correct_answer,
        train_formatters=FormatterOptions.few_shot,
    ),
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:far-ai::8CFH68SY",
        trained_samples=50000,
        filter_strategy=FilterStrategy.correct_answer,
        train_formatters=FormatterOptions.few_shot,
    ),
]
ZERO_SHOT = [
    # trained on zero shot biases
    ModelTrainMeta(
        name="gpt-3.5-turbo",
        trained_samples=1,
        filter_strategy=FilterStrategy.correct_answer,
        train_formatters=FormatterOptions.zero_shot,
    ),
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:far-ai::8BRpCYNt",
        trained_samples=100,
        filter_strategy=FilterStrategy.correct_answer,
        train_formatters=FormatterOptions.zero_shot,
    ),
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:far-ai::8BSJekFR",
        trained_samples=1000,
        filter_strategy=FilterStrategy.correct_answer,
        train_formatters=FormatterOptions.zero_shot,
    ),
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:far-ai::8BSeBItZ",
        trained_samples=10000,
        filter_strategy=FilterStrategy.correct_answer,
        train_formatters=FormatterOptions.zero_shot,
    ),
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:far-ai::8BSkM7rh",
        trained_samples=20000,
        filter_strategy=FilterStrategy.correct_answer,
        train_formatters=FormatterOptions.zero_shot,
    ),
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:far-ai::8CCOhcca",
        trained_samples=50000,
        filter_strategy=FilterStrategy.correct_answer,
        train_formatters=FormatterOptions.zero_shot,
    ),
]
OG_CONTROL = [
    ModelTrainMeta(
        name="gpt-3.5-turbo",
        trained_samples=1,
        filter_strategy=FilterStrategy.correct_answer,
        train_formatters=FormatterOptions.control_only_unbiased,
    ),
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:academicsnyuperez::89NHOL5b",
        trained_samples=100,
        filter_strategy=FilterStrategy.correct_answer,
        train_formatters=FormatterOptions.control_only_unbiased,
    ),
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:academicsnyuperez::89G2vwHZ",
        trained_samples=1000,
        filter_strategy=FilterStrategy.correct_answer,
        train_formatters=FormatterOptions.control_only_unbiased,
    ),
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:academicsnyuperez::89GzBGx0",
        trained_samples=10000,
        filter_strategy=FilterStrategy.correct_answer,
        train_formatters=FormatterOptions.control_only_unbiased,
    ),
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:academicsnyuperez::89LJSEdM",
        trained_samples=20000,
        filter_strategy=FilterStrategy.correct_answer,
        train_formatters=FormatterOptions.control_only_unbiased,
    ),
]