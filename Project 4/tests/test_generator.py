
import pytest
from app import app
from src.preprocessing import clean_text, parse_structured_input, build_generation_prompt
from src.evaluation import get_text_statistics, calculate_evaluation_metrics


def test_clean_text():
    assert clean_text("  Hello   World!  \n") == "Hello World!"

    with pytest.raises(ValueError):
        clean_text("")

    with pytest.raises(ValueError):
        clean_text("   \n\t  ")

    with pytest.raises(ValueError):
        clean_text(None)


def test_parse_structured_input():
    kv_str = "Department: Computer Science\nStudents: 120\nPass Percentage: 94%"
    parsed_str = parse_structured_input(kv_str)
    assert "Department: Computer Science" in parsed_str
    assert "Students: 120" in parsed_str

    dict_input = {"Revenue": "₹12.5M", "Growth": "18%"}
    parsed_dict = parse_structured_input(dict_input)
    assert "Revenue: ₹12.5M" in parsed_dict
    assert "Growth: 18%" in parsed_dict

    json_str = '{"Customers": 24500, "Churn": "4.2%"}'
    parsed_json = parse_structured_input(json_str)
    assert "Customers: 24500" in parsed_json


def test_build_generation_prompt():
    prompt = build_generation_prompt(
        prompt_text="Write a short summary.",
        structured_data="Department: CS",
        style="formal"
    )
    assert "formal" in prompt.lower() or "professional" in prompt.lower()
    assert "Department: CS" in prompt
    assert "Write a short summary." in prompt

    concise_prompt = build_generation_prompt(
        prompt_text="Explain project highlights.",
        style="concise"
    )
    assert "concisely" in concise_prompt.lower() or "short" in concise_prompt.lower()

    with pytest.raises(ValueError):
        build_generation_prompt(prompt_text=None, structured_data=None)


def test_get_text_statistics():
    sample_text = "The Computer Science department offers great education. Students excel in Data Structures."
    stats = get_text_statistics(sample_text)

    assert stats["word_count"] > 10
    assert stats["sentence_count"] == 2
    assert stats["char_count"] == len(sample_text)
    assert stats["avg_word_length"] > 0.0


def test_rouge_metrics_exact_and_partial():
    ref_identical = "The Electronics department demonstrates strong academic performance among its 245 students."
    gen_identical = "The Electronics department demonstrates strong academic performance among its 245 students."

    res_identical = calculate_evaluation_metrics(gen_identical, ref_identical)
    assert res_identical["metrics_available"] is True
    rouge_exact = res_identical["metrics"]["rouge"]
    assert rouge_exact["rouge1"]["fmeasure"] >= 0.99
    assert rouge_exact["rouge2"]["fmeasure"] >= 0.99
    assert rouge_exact["rougeL"]["fmeasure"] >= 0.99

    ref_partial = "The Electronics department demonstrates strong academic performance among its 245 students."
    gen_partial = "The Department of Electronics has 245 students."

    res_partial = calculate_evaluation_metrics(gen_partial, ref_partial)
    assert res_partial["metrics_available"] is True
    rouge_part = res_partial["metrics"]["rouge"]
    assert rouge_part["rouge1"]["fmeasure"] > 0.0
    assert rouge_part["rouge2"]["fmeasure"] > 0.0
    assert rouge_part["rougeL"]["fmeasure"] > 0.0


def test_rouge_metrics_unrelated():
    ref = "Quantum computing uses qubits."
    gen = "Baking bread requires flour."
    res = calculate_evaluation_metrics(gen, ref)
    assert res["metrics_available"] is True
    rouge = res["metrics"]["rouge"]
    assert rouge["rouge1"]["fmeasure"] == 0.0
    assert rouge["rouge2"]["fmeasure"] == 0.0
    assert rouge["rougeL"]["fmeasure"] == 0.0


def test_rouge_metrics_empty_input():
    res_empty_ref = calculate_evaluation_metrics("Sample generated text.", "")
    assert res_empty_ref["metrics_available"] is False
    assert res_empty_ref["metrics"] is None

    res_empty_gen = calculate_evaluation_metrics("", "Sample reference text.")
    assert res_empty_gen["metrics_available"] is True
    assert res_empty_gen["metrics"]["rouge"]["rouge1"]["fmeasure"] == 0.0


def test_fact_coverage_all_facts_present():
    from src.preprocessing import extract_structured_facts
    from src.generator import check_fact_coverage

    struct_data = """Department: Electronics
Students: 245
Average Attendance: 87%
Pass Percentage: 91%
Top Subject: Digital Electronics
Placement Rate: 82%
Faculty Members: 24"""

    facts = extract_structured_facts(struct_data)
    full_text = "The Department of Electronics has 245 students, an average attendance of 87%, a pass percentage of 91%, a placement rate of 82%, and 24 faculty members. The top subject is Digital Electronics."

    cov = check_fact_coverage(facts, full_text)
    assert cov["complete"] is True
    assert cov["covered"] is True
    assert cov["coverage"] == "7/7"
    assert len(cov["missing_facts"]) == 0


def test_fact_coverage_missing_percentage():
    from src.preprocessing import extract_structured_facts
    from src.generator import check_fact_coverage

    struct_data = """Department: Electronics
Students: 245
Average Attendance: 87%
Pass Percentage: 91%
Top Subject: Digital Electronics
Placement Rate: 82%
Faculty Members: 24"""

    facts = extract_structured_facts(struct_data)
    text_missing_82 = "The Department of Electronics has 245 students, an average attendance of 87%, a pass percentage of 91%, and 24 faculty members. The top subject is Digital Electronics."

    cov = check_fact_coverage(facts, text_missing_82)
    assert cov["complete"] is False
    assert cov["coverage"] == "6/7"
    assert "82%" in cov["missing_facts"]


def test_generation_retries_when_fact_missing():
    from src.generator import NLGGenerator
    gen = NLGGenerator.get_instance()

    struct_data = """Department: Electronics
Students: 245
Average Attendance: 87%
Pass Percentage: 91%
Top Subject: Digital Electronics
Placement Rate: 82%
Faculty Members: 24"""

    result = gen.generate(
        prompt_text="Write a formal report",
        structured_data=struct_data,
        style="formal"
    )
    assert "generated_text" in result
    assert result["fact_coverage"]["complete"] is True
    assert result["fact_coverage"]["coverage"] == "7/7"
    assert "82%" in result["generated_text"]


def test_extract_structured_facts_key_value_pairs():
    from src.preprocessing import extract_structured_facts
    struct_data = """Department: Artificial Intelligence
Students: 1,250
Pass Percentage: 96.25%
Placement Rate: 92.75%"""

    facts = extract_structured_facts(struct_data)
    assert len(facts) == 4
    assert facts[0]["key"] == "Department" and facts[0]["value"] == "Artificial Intelligence"
    assert facts[1]["key"] == "Students" and facts[1]["value"] == "1,250"
    assert facts[2]["key"] == "Pass Percentage" and facts[2]["value"] == "96.25%"
    assert facts[3]["key"] == "Placement Rate" and facts[3]["value"] == "92.75%"


def test_semantic_fact_coverage_percentages_and_prevention_of_malformed_output():
    from src.preprocessing import extract_structured_facts
    from src.generator import check_fact_coverage, format_semantic_safeguard_clause

    struct_data = """Pass Percentage: 96.25%
Placement Rate: 92.75%"""

    facts = extract_structured_facts(struct_data)

    bad_text = "The department has a placement rate of 92.75% and rate is 96.25%."
    cov = check_fact_coverage(facts, bad_text)
    assert cov["complete"] is False

    good_text = "The department achieves a pass percentage of 96.25% and a placement rate of 92.75%."
    cov_good = check_fact_coverage(facts, good_text)
    assert cov_good["complete"] is True

    c1 = format_semantic_safeguard_clause(facts[0])
    c2 = format_semantic_safeguard_clause(facts[1])
    assert "pass percentage of 96.25%" in c1
    assert "placement rate of 92.75%" in c2
    assert "rate is 96.25%" not in c1


def test_regression_malformed_string_rejected():
    from src.preprocessing import extract_structured_facts
    from src.generator import check_fact_coverage

    struct_data = """Pass Percentage: 96.25%
Placement Rate: 92.75%"""

    facts = extract_structured_facts(struct_data)
    malformed_output = "The average attendance is 89.5%. Additionally, the report notes 1,250, rate is 96.25%, rate is 92.75%."

    cov = check_fact_coverage(facts, malformed_output)
    assert cov["semantic_coverage"] is False
    assert cov["complete"] is False


def test_regression_correct_semantic_output_accepted():
    from src.preprocessing import extract_structured_facts
    from src.generator import check_fact_coverage

    struct_data = """Pass Percentage: 96.25%
Placement Rate: 92.75%"""

    facts = extract_structured_facts(struct_data)
    correct_output = "The pass percentage is 96.25% and the placement rate is 92.75%."

    cov = check_fact_coverage(facts, correct_output)
    assert cov["semantic_coverage"] is True
    assert cov["complete"] is True
    assert len(cov["fact_details"]) == 2
    assert cov["fact_details"][0]["semantic_match"] is True
    assert cov["fact_details"][1]["semantic_match"] is True


def test_regression_safeguard_clauses_for_percentages_and_students():
    from src.generator import format_semantic_safeguard_clause

    f_pass = {"key": "Pass Percentage", "value": "96.25%", "raw_str": "Pass Percentage: 96.25%"}
    c_pass = format_semantic_safeguard_clause(f_pass)
    assert "pass percentage" in c_pass
    assert "96.25%" in c_pass

    f_place = {"key": "Placement Rate", "value": "92.75%", "raw_str": "Placement Rate: 92.75%"}
    c_place = format_semantic_safeguard_clause(f_place)
    assert "placement rate" in c_place
    assert "92.75%" in c_place

    f_students = {"key": "Students", "value": "1,250", "raw_str": "Students: 1,250"}
    c_students = format_semantic_safeguard_clause(f_students)
    assert "students" in c_students
    assert "1,250" in c_students


def test_generalization_company_dataset():
    from src.preprocessing import extract_structured_facts
    from src.generator import check_fact_coverage

    struct_data = """Revenue: $2.5 million
Employees: 145
Customer Satisfaction: 91%
Founded: 2018"""

    facts = extract_structured_facts(struct_data)
    text = "The company recorded revenue of $2.5 million with 145 employees and a customer satisfaction rate of 91%. It was founded in 2018."
    cov = check_fact_coverage(facts, text)
    assert cov["complete"] is True
    assert cov["semantic_coverage"] is True
    assert cov["coverage"] == "4/4"


def test_generalization_university_dataset():
    from src.preprocessing import extract_structured_facts
    from src.generator import check_fact_coverage

    struct_data = """Students: 5000
Graduation Rate: 88.5%
Departments: 12
Established: 1998"""

    facts = extract_structured_facts(struct_data)
    text = "The university serves 5000 students across 12 departments with a graduation rate of 88.5%. The institution was established in 1998."
    cov = check_fact_coverage(facts, text)
    assert cov["complete"] is True
    assert cov["semantic_coverage"] is True
    assert cov["coverage"] == "4/4"


def test_generalization_decimal_values_dataset():
    from src.preprocessing import extract_structured_facts
    from src.generator import check_fact_coverage

    struct_data = """Accuracy: 96.25%
Latency: 12.75 ms
Version: 2.5"""

    facts = extract_structured_facts(struct_data)
    text = "The system version 2.5 achieved an accuracy of 96.25% with a latency of 12.75 ms."
    cov = check_fact_coverage(facts, text)
    assert cov["complete"] is True
    assert cov["semantic_coverage"] is True
    assert cov["coverage"] == "3/3"


def test_generalization_comma_formatted_numbers():
    from src.preprocessing import extract_structured_facts
    from src.generator import check_fact_coverage

    struct_data = """Revenue: 1,250,000
Students: 12,500"""

    facts = extract_structured_facts(struct_data)
    text = "The organization reached revenue of 1,250,000 and enrolled 12,500 students."
    cov = check_fact_coverage(facts, text)
    assert cov["complete"] is True
    assert cov["semantic_coverage"] is True
    assert cov["coverage"] == "2/2"


def test_malformed_generation_rejection_requirement():
    from src.preprocessing import extract_structured_facts
    from src.generator import check_fact_coverage

    struct_data = """Pass Percentage: 96.25%
Placement Rate: 92.75%
Students: 1,250"""

    facts = extract_structured_facts(struct_data)
    malformed_text = "The department has 1,250, rate is 96.25%, rate is 92.75%."
    cov = check_fact_coverage(facts, malformed_text)
    assert cov["complete"] is False
    assert cov["semantic_coverage"] is False


def test_grammar_agreement_and_malformed_headquarters_rejection():
    from src.preprocessing import extract_structured_facts
    from src.generator import check_fact_coverage

    struct_data = """Company: NovaTech Solutions
Headquarters: Hyderabad
Customer Satisfaction: 91%"""

    facts = extract_structured_facts(struct_data)

    bad_text = "Its headquarters are located in Hyderabad and has a customer satisfaction of 91%."
    cov_bad = check_fact_coverage(facts, bad_text)
    assert cov_bad["complete"] is False

    good_text = "NovaTech Solutions has headquarters located in Hyderabad, and the company has a customer satisfaction rate of 91%."
    cov_good = check_fact_coverage(facts, good_text)
    assert cov_good["complete"] is True


def test_natural_verb_phrasing_in_safeguard_clauses():
    from src.generator import format_semantic_safeguard_clause

    f_dept = {"key": "Departments", "value": "12"}
    c_dept = format_semantic_safeguard_clause(f_dept)
    assert "has 12 departments" in c_dept

    f_ver = {"key": "Version", "value": "2.5"}
    c_ver = format_semantic_safeguard_clause(f_ver)
    assert "uses version 2.5" in c_ver

    f_grad = {"key": "Graduation Rate", "value": "88.5%"}
    c_grad = format_semantic_safeguard_clause(f_grad)
    assert "has a graduation rate of 88.5%" in c_grad


def test_double_verb_shows_uses_rejection_and_clean_domain_neutral_formatting():
    from src.preprocessing import extract_structured_facts
    from src.generator import check_fact_coverage

    struct_data = """Accuracy: 96.25%
Latency: 12.75 ms
Version: 2.5"""

    facts = extract_structured_facts(struct_data)

    bad_text = "The accuracy was 96.25% and the latency was 12.75 ms. Additionally, the report shows uses version 2.5."
    cov_bad = check_fact_coverage(facts, bad_text)
    assert cov_bad["semantic_coverage"] is False
    assert cov_bad["complete"] is False

    good_text = "The accuracy was 96.25% and the latency was 12.75 ms. Additionally, the report uses version 2.5."
    cov_good = check_fact_coverage(facts, good_text)
    assert cov_good["semantic_coverage"] is True
    assert cov_good["complete"] is True
    assert cov_good["coverage"] == "3/3"


def test_greengrid_initiative_end_to_end():
    client = app.test_client()

    struct_data = """Project: GreenGrid Initiative
Regions: 27
Annual Savings: $8.4 million
Emission Reduction: 37.8%
Implementation Time: 14.5 months
Completion Year: 2026"""

    res = client.post("/generate", json={
        "prompt_text": "Write a project report for the GreenGrid Initiative using the provided information.",
        "structured_data": struct_data,
        "style": "formal"
    })
    data = res.get_json()
    assert data["success"] is True
    out_text = data["result"]["generated_text"]
    fc = data["fact_coverage"]

    assert fc["semantic_coverage"] is True
    assert fc["coverage"] == "6/6"

    assert "GreenGrid Initiative" in out_text
    assert "the report has" not in out_text.lower()
    assert "the report uses" not in out_text.lower()

    assert "has implementation time of" not in out_text.lower()
    assert "2026 completion year" not in out_text.lower()
    assert "shows uses" not in out_text.lower()


def test_unrelated_project_dataset():
    client = app.test_client()

    struct_data = """Initiative: Urban Transit 2030
Coverage Area: 15 districts
Budget: $4.2 million
Turnaround Time: 18 months
Target Year: 2028"""

    res = client.post("/generate", json={
        "prompt_text": "Write a summary report for Urban Transit 2030.",
        "structured_data": struct_data,
        "style": "formal"
    })
    data = res.get_json()
    assert data["success"] is True
    fc = data["fact_coverage"]
    assert fc["semantic_coverage"] is True
    assert fc["coverage"] == "5/5"


def test_company_dataset_generalization():
    client = app.test_client()

    struct_data = """Company: SolarMax Dynamics
Revenue: $12.5 million
Employees: 320
Headquarters: Austin
Founded: 2015"""

    res = client.post("/generate", json={
        "prompt_text": "Write a company profile for SolarMax Dynamics.",
        "structured_data": struct_data,
        "style": "formal"
    })
    data = res.get_json()
    assert data["success"] is True
    fc = data["fact_coverage"]
    assert fc["semantic_coverage"] is True
    assert fc["coverage"] == "5/5"


def test_technical_system_dataset_generalization():
    client = app.test_client()

    struct_data = """System: DataEngine X
Accuracy: 99.1%
Processing Duration: 4.5 seconds
Version: 4.2"""

    res = client.post("/generate", json={
        "prompt_text": "Write a performance report for DataEngine X.",
        "structured_data": struct_data,
        "style": "formal"
    })
    data = res.get_json()
    assert data["success"] is True
    fc = data["fact_coverage"]
    assert fc["semantic_coverage"] is True
    assert fc["coverage"] == "4/4"


def test_unseen_domain_key_dataset_generalization():
    client = app.test_client()

    struct_data = """App: HealthPulse
Storage Capacity: 50 GB
Active Subscribers: 85,000
Satisfaction Index: 94.5%"""

    res = client.post("/generate", json={
        "prompt_text": "Write a product report for HealthPulse.",
        "structured_data": struct_data,
        "style": "formal"
    })
    data = res.get_json()
    assert data["success"] is True
    fc = data["fact_coverage"]
    assert fc["semantic_coverage"] is True
    assert fc["coverage"] == "4/4"


def test_semantic_category_classification_unit():
    from src.generator import classify_semantic_category, SemanticCategory, format_semantic_safeguard_clause

    assert classify_semantic_category("Energy Efficiency", "93.7%") == SemanticCategory.PERCENTAGE
    c_perc = format_semantic_safeguard_clause({"key": "Energy Efficiency", "value": "93.7%"})
    assert c_perc == "has an energy efficiency of 93.7%"

    assert classify_semantic_category("Growth Rate", "14.8%") == SemanticCategory.PERCENTAGE

    assert classify_semantic_category("Annual Budget", "$5.5 million") == SemanticCategory.CURRENCY
    c_curr = format_semantic_safeguard_clause({"key": "Annual Budget", "value": "$5.5 million"})
    assert c_curr == "has an annual budget of $5.5 million"

    assert classify_semantic_category("Processing Time", "8.5 seconds") == SemanticCategory.DURATION
    c_dur = format_semantic_safeguard_clause({"key": "Processing Time", "value": "8.5 seconds"})
    assert c_dur == "has a processing time of 8.5 seconds"

    assert classify_semantic_category("Release Year", "2027") == SemanticCategory.YEAR
    c_yr = format_semantic_safeguard_clause({"key": "Release Year", "value": "2027"})
    assert c_yr == "has a release year of 2027"

    assert classify_semantic_category("Headquarters", "Berlin") == SemanticCategory.LOCATION
    c_loc = format_semantic_safeguard_clause({"key": "Headquarters", "value": "Berlin"})
    assert c_loc == "is headquartered in Berlin"

    assert classify_semantic_category("System Version", "3.0") == SemanticCategory.VERSION
    c_ver = format_semantic_safeguard_clause({"key": "System Version", "value": "3.0"})
    assert c_ver == "uses version 3.0"

    assert classify_semantic_category("Processing Capacity", "72,500 requests/sec") == SemanticCategory.CAPACITY
    c_cap = format_semantic_safeguard_clause({"key": "Processing Capacity", "value": "72,500 requests/sec"})
    assert c_cap == "has a processing capacity of 72,500 requests/sec"

    assert classify_semantic_category("Active Subscribers", "45,000") == SemanticCategory.COUNT
    c_cnt = format_semantic_safeguard_clause({"key": "Active Subscribers", "value": "45,000"})
    assert c_cnt == "has 45,000 active subscribers"

    assert classify_semantic_category("Arbitrary Metric", "100") == SemanticCategory.COUNT
    c_oth = format_semantic_safeguard_clause({"key": "Arbitrary Metric", "value": "100"})
    assert c_oth == "has an arbitrary metric of 100"


def test_completely_unseen_quantumflow_platform_dataset():
    client = app.test_client()

    struct_data = """Entity: QuantumFlow Platform
Coverage Area: 185 km²
Processing Capacity: 72,500 requests/sec
Energy Efficiency: 93.7%
Deployment Time: 11.5 months
Release Year: 2027"""

    res = client.post("/generate", json={
        "prompt_text": "Write a performance overview for QuantumFlow Platform.",
        "structured_data": struct_data,
        "style": "formal"
    })

    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True

    out_text = data["result"]["generated_text"]
    fc = data["fact_coverage"]

    assert fc["semantic_coverage"] is True
    assert fc["coverage"] == "6/6"

    assert "QuantumFlow Platform" in out_text
    assert "the report has" not in out_text.lower()
    assert "the report uses" not in out_text.lower()

    assert "185" in out_text
    assert "72,500" in out_text
    assert "93.7%" in out_text
    assert "11.5" in out_text
    assert "2027" in out_text

    assert "shows uses" not in out_text.lower()
    assert "2027 release year" not in out_text.lower()


def test_end_to_end_ai_department_dataset():
    client = app.test_client()

    struct_data = """Department: Artificial Intelligence
Students: 1,250
Average Attendance: 89.5%
Pass Percentage: 96.25%
Top Subject: Machine Learning
Placement Rate: 92.75%
Faculty Members: 32
Established: 2021"""

    ref_text = "The Artificial Intelligence department has 1,250 students and maintains an average attendance of 89.5%. The department achieves a pass percentage of 96.25% and a placement rate of 92.75%. Machine Learning is the top subject, supported by 32 faculty members. The department was established in 2021."

    res = client.post("/generate", json={
        "prompt_text": "Write a formal performance report for the Artificial Intelligence department using the provided information.",
        "structured_data": struct_data,
        "reference_text": ref_text,
        "style": "formal"
    })

    assert res.status_code == 200
    json_data = res.get_json()
    assert json_data["success"] is True

    out_text = json_data["result"]["generated_text"]
    fc = json_data["fact_coverage"]

    assert fc["complete"] is True
    assert fc["semantic_coverage"] is True
    assert fc["coverage"] == "8/8"
    assert "fact_details" in fc
    assert len(fc["fact_details"]) == 8

    assert "96.25%" in out_text
    assert "92.75%" in out_text

    assert "rate is 96.25%" not in out_text
    assert "1,250, rate is" not in out_text

    metrics = json_data["evaluation"]["metrics"]
    assert metrics["rouge"]["rouge1"]["fmeasure"] > 0
    assert metrics["rouge"]["rouge2"]["fmeasure"] > 0
    assert metrics["rouge"]["rougeL"]["fmeasure"] > 0
    assert metrics["bleu"]["bleu_1"] > 0


from unittest.mock import MagicMock, patch

def test_flask_api_routes():
    mock_gen = MagicMock()
    mock_gen.model_name = "google/flan-t5-base"
    mock_gen.device = "cpu"
    mock_gen.generate.return_value = {
        "generated_text": "The CS department has 120 students.",
        "full_prompt": "Generate text for CS department",
        "prompt_text": "CS dept prompt",
        "structured_data": None,
        "style": "general",
        "model_name": "google/flan-t5-base",
        "device": "cpu",
        "execution_time_sec": 0.25,
        "parameters": {}
    }

    with patch("app.get_generator", return_value=mock_gen):
        client = app.test_client()

        response = client.get("/")
        assert response.status_code == 200
        assert b"Natural Language Generator" in response.data

        health_resp = client.get("/health")
        assert health_resp.status_code == 200
        health_json = health_resp.get_json()
        assert health_json["status"] == "healthy"
        assert health_json["model_name"] == "google/flan-t5-base"

        gen_resp = client.post("/generate", json={
            "prompt_text": "Write a short summary",
            "style": "general"
        })
        assert gen_resp.status_code == 200
        gen_json = gen_resp.get_json()
        assert gen_json["success"] is True
        assert "result" in gen_json

        eval_resp = client.post("/evaluate", json={
            "generated_text": "Sample generated text output.",
            "reference_text": "Sample reference ground-truth text."
        })
        assert eval_resp.status_code == 200
        eval_json = eval_resp.get_json()
        assert eval_json["success"] is True

        bad_resp = client.post("/generate", json={})
        assert bad_resp.status_code == 400

        clear_resp = client.post("/clear")
        assert clear_resp.status_code == 200
        assert clear_resp.get_json()["success"] is True


def test_semantic_substitution_rejection_energy_efficiency():
    from src.preprocessing import extract_structured_facts
    from src.generator import check_fact_coverage

    struct_data = "Energy Efficiency: 93.7%"
    facts = extract_structured_facts(struct_data)
    text = "The platform features a power efficiency of 93.7%."
    cov = check_fact_coverage(facts, text)
    assert cov["semantic_coverage"] is False
    assert cov["complete"] is False
    assert cov["fact_details"][0]["semantic_match"] is False


def test_coverage_area_semantic_substitution_rejection():
    from src.preprocessing import extract_structured_facts
    from src.generator import check_fact_coverage

    struct_data = "Coverage Area: 185 km²"
    facts = extract_structured_facts(struct_data)
    text = "The system has a total area of 185 km²."
    cov = check_fact_coverage(facts, text)
    assert cov["semantic_coverage"] is False
    assert cov["complete"] is False
    assert cov["fact_details"][0]["semantic_match"] is False


def test_unsupported_event_inference_rejection():
    from src.preprocessing import extract_structured_facts
    from src.generator import check_fact_coverage

    struct_data = "Release Year: 2027"
    facts = extract_structured_facts(struct_data)
    text = "The platform was launched in 2027."
    cov = check_fact_coverage(facts, text)
    assert cov["semantic_coverage"] is False
    assert cov["complete"] is False
    assert cov["fact_details"][0]["semantic_match"] is False


def test_semantic_equivalent_acceptance():
    from src.preprocessing import extract_structured_facts
    from src.generator import check_fact_coverage

    struct_data = "Coverage Area: 185 km²"
    facts = extract_structured_facts(struct_data)
    text = "The system covers 185 km² efficiently."
    cov = check_fact_coverage(facts, text)
    assert cov["semantic_coverage"] is True
    assert cov["complete"] is True
    assert cov["fact_details"][0]["semantic_match"] is True


def test_safeguard_deduplication():
    from src.preprocessing import extract_structured_facts
    from src.generator import check_fact_coverage, NLGGenerator

    struct_data = """Entity: QuantumFlow Platform
Coverage Area: 185 km²
Processing Capacity: 72,500 requests/sec
Energy Efficiency: 93.7%
Deployment Time: 11.5 months
Release Year: 2027"""

    gen = NLGGenerator.get_instance()
    res = gen.generate(
        prompt_text="Summarize QuantumFlow Platform performance.",
        structured_data=struct_data,
        style="formal"
    )

    out_text = res["generated_text"]
    cov = res["fact_coverage"]

    assert cov["semantic_coverage"] is True
    assert cov["coverage"] == "6/6"

    out_lower = out_text.lower()
    assert out_lower.count("185 km²") <= 1 or out_lower.count("covers 185") <= 1
    assert out_lower.count("2027") == 1
    assert out_lower.count("11.5 months") == 1
    assert out_lower.count("93.7%") == 1


def test_partial_safeguard():
    from src.preprocessing import extract_structured_facts
    from src.generator import check_fact_coverage, format_semantic_safeguard_clause

    struct_data = """Entity: QuantumFlow Platform
Coverage Area: 185 km²
Processing Capacity: 72,500 requests/sec
Energy Efficiency: 93.7%
Deployment Time: 11.5 months
Release Year: 2027"""

    facts = extract_structured_facts(struct_data)
    partial_text = "QuantumFlow Platform covers 185 km² and has a processing capacity of 72,500 requests/sec. Its energy efficiency is 93.7%."

    cov = check_fact_coverage(facts, partial_text)
    assert cov["complete"] is False
    assert cov["covered_count"] == 4

    missing_fact_dicts = [f for f in facts if f["value"] in cov["missing_facts"]]
    assert len(missing_fact_dicts) == 2

    clauses = [format_semantic_safeguard_clause(fd) for fd in missing_fact_dicts]
    assert any("deployment time" in c for c in clauses)
    assert any("release year" in c for c in clauses)
    assert not any("coverage area" in c for c in clauses)


def test_completely_unseen_aquapure_system_dataset():
    client = app.test_client()

    struct_data = """Entity: AquaPure System
Filtration Capacity: 4,200 liters/hr
Purity Index: 99.8%
Operating Duration: 3.5 years
Certification Year: 2029"""

    res = client.post("/generate", json={
        "prompt_text": "Write an overview report for AquaPure System.",
        "structured_data": struct_data,
        "style": "formal"
    })

    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True

    out_text = data["result"]["generated_text"]
    fc = data["fact_coverage"]

    assert fc["semantic_coverage"] is True
    assert fc["coverage"] == "5/5"
    assert "AquaPure System" in out_text
    assert "4,200" in out_text
    assert "99.8%" in out_text
    assert "3.5" in out_text
    assert "2029" in out_text


def test_four_generation_styles_semantic_fidelity():
    client = app.test_client()

    struct_data = """Entity: QuantumFlow Platform
Coverage Area: 185 km²
Processing Capacity: 72,500 requests/sec
Energy Efficiency: 93.7%
Deployment Time: 11.5 months
Release Year: 2027"""

    styles = ["general", "formal", "concise", "descriptive"]

    for style in styles:
        res = client.post("/generate", json={
            "prompt_text": f"Generate a summary for QuantumFlow Platform in {style} style.",
            "structured_data": struct_data,
            "style": style
        })

        assert res.status_code == 200, f"Failed for style {style}"
        data = res.get_json()
        assert data["success"] is True, f"Failed success for style {style}"

        out_text = data["result"]["generated_text"]
        fc = data["fact_coverage"]

        assert fc["semantic_coverage"] is True, f"Style {style} failed semantic coverage"
        assert fc["coverage"] == "6/6", f"Style {style} expected 6/6 but got {fc['coverage']}"

        out_lower = out_text.lower()
        assert "power efficiency" not in out_lower, f"Style {style} contained power efficiency"
        assert "total area" not in out_lower, f"Style {style} contained total area"

        if "launched" in out_lower:
            assert "release" in out_lower or "release year" in out_lower, f"Style {style} introduced launched without release key"

        assert "185" in out_text
        assert "72,500" in out_text
        assert "93.7%" in out_text
        assert "11.5" in out_text
        assert "2027" in out_text


def test_aquapure_system_four_styles_semantic_fidelity():
    client = app.test_client()

    struct_data = """Entity: AquaPure System
Filtration Capacity: 48,600 liters/hour
Water Purity: 98.4%
Coverage Area: 240 km²
Operating Duration: 16 months
Release Year: 2028"""

    styles = ["general", "formal", "concise", "descriptive"]

    for style in styles:
        res = client.post("/generate", json={
            "prompt_text": f"Write an overview report for AquaPure System in {style} style.",
            "structured_data": struct_data,
            "style": style
        })

        assert res.status_code == 200, f"Failed HTTP status for style {style}"
        data = res.get_json()
        assert data["success"] is True, f"Failed success for style {style}"

        out_text = data["result"]["generated_text"]
        fc = data["fact_coverage"]

        assert fc["semantic_coverage"] is True, f"Style {style} failed semantic coverage"
        assert fc["coverage"] == "6/6", f"Style {style} expected 6/6 coverage but got {fc['coverage']}"

        out_lower = out_text.lower()

        assert "was launched" not in out_lower, f"Style {style} introduced unsupported event 'was launched'"
        assert "was released" not in out_lower, f"Style {style} introduced unsupported event 'was released'"
        assert "was introduced" not in out_lower, f"Style {style} introduced unsupported event 'was introduced'"
        assert "scheduled to be" not in out_lower, f"Style {style} introduced modal schedule inference 'scheduled to be'"
        assert "expected to be" not in out_lower, f"Style {style} introduced modal schedule inference 'expected to be'"

        assert "power efficiency" not in out_lower, f"Style {style} substituted Water Purity with power efficiency"
        assert "total area" not in out_lower, f"Style {style} substituted Coverage Area with total area"
        assert "cover area" not in out_lower, f"Style {style} used malformed label cover area"
        assert "processing capacity" not in out_lower, f"Style {style} substituted Filtration Capacity with processing capacity"
        assert "water purification system" not in out_lower, f"Style {style} inferred unprovided entity type water purification system"

        assert out_lower.count("coverage area") <= 1, f"Style {style} duplicated coverage area fact"
        assert out_lower.count("water purity") <= 1, f"Style {style} duplicated water purity fact"
        assert out_lower.count("filtration capacity") <= 1, f"Style {style} duplicated filtration capacity fact"

        assert "AquaPure System" in out_text, f"Style {style} lost entity name"
        assert "48,600" in out_text, f"Style {style} lost 48,600 value"
        assert "98.4%" in out_text, f"Style {style} lost 98.4% value"
        assert "240" in out_text, f"Style {style} lost 240 value"
        assert "16" in out_text, f"Style {style} lost 16 value"
        assert "2028" in out_text, f"Style {style} lost 2028 value"


def test_regression_unsupported_modal_schedule_inference_rejected():
    from src.generator import check_fact_coverage
    facts = [{"key": "Release Year", "value": "2028"}]
    gen_text = "AquaPure System is scheduled to be released in 2028."
    cov = check_fact_coverage(facts, gen_text)
    assert cov["semantic_coverage"] is False, "Failed to reject modal schedule inference 'scheduled to be released'"


def test_regression_malformed_cover_area_label_rejected():
    from src.generator import check_fact_coverage
    facts = [{"key": "Coverage Area", "value": "240 km²"}]
    gen_text = "AquaPure System has a cover area of 240 km²."
    cov = check_fact_coverage(facts, gen_text)
    assert cov["semantic_coverage"] is False, "Failed to reject malformed label 'cover area'"


def test_regression_unprovided_entity_type_purification_rejected():
    from src.generator import check_fact_coverage
    facts = [{"key": "Filtration Capacity", "value": "48,600 liters/hour"}]
    gen_text = "AquaPure System is a water purification system with a capacity of 48,600 liters/hour."
    cov = check_fact_coverage(facts, gen_text)
    assert cov["semantic_coverage"] is False, "Failed to reject 'water purification system' for Filtration Capacity"


