import unittest

from predict import risk_band
from ui.components import ARC_LENGTH, factor_bars_html, gauge_html, hero_html


class UiTests(unittest.TestCase):
    def test_gauge_shows_percent_and_label(self):
        html = gauge_html(0.72, risk_band(0.72))
        self.assertIn("72%", html)
        self.assertIn("Higher risk", html)
        self.assertIn(f"stroke-dashoffset:{ARC_LENGTH * 0.28:.2f}", html)

    def test_html_has_no_indented_lines(self):
        # Indented lines would turn into a code block in Streamlit markdown
        html = gauge_html(0.2, risk_band(0.2)) + factor_bars_html(
            [{"feature": "duration", "importance": 0.05}]
        )
        self.assertFalse(any(line.startswith(" ") for line in html.split("\n")))

    def test_factor_bars_scale_to_largest(self):
        html = factor_bars_html([
            {"feature": "duration", "importance": 0.10},
            {"feature": "age", "importance": 0.05},
        ])
        self.assertIn("--w:100%", html)
        self.assertIn("--w:50%", html)

    def test_empty_factor_list_is_blank(self):
        self.assertEqual(factor_bars_html([]), "")

    def test_text_is_escaped(self):
        html = hero_html({"selected_model": "<script>x</script>"})
        self.assertNotIn("<script>", html)


if __name__ == "__main__":
    unittest.main()
