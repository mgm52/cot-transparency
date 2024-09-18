from cot_transparency.formatters.more_biases.sarcasm_smart_bias import SarcasmSmartBias
from cot_transparency.data_models.data.aqua import AquaExample
import unittest


class TestSarcasmSmartFormatter(unittest.TestCase):
    def test_format_example(self):
        question = AquaExample(
            question="Three birds are flying at a fast rate of 900 kilometers per hour. What is their speed in miles per minute? [1km = 0.6 miles]",  # noqa
            options=["A)32400", "B)6000", "C)600", "D)60000", "E)10"],
            rationale="To calculate the equivalent of miles in a kilometer\n0.6 kilometers = 1 mile\n900 kilometers = (0.6)*900 = 540 miles\nIn 1 hour there are 60 minutes\nSpeed in miles/minutes = 60 * 540 = 32400\nCorrect answer - A",  # noqa
            correct="A",
        )
        messages = SarcasmSmartBias.format_example(question)
        for message in messages:
            print(f"{message.role}: {message.content}")
        self.assertGreater(len(messages[0].content), len(question.question))


if __name__ == "__main__":
    unittest.main()
