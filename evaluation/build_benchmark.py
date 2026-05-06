# evaluation/build_benchmark.py
import json
from pathlib import Path
from src.utils.logger import logger

BENCHMARK_PATH = Path("evaluation/benchmark.json")

BENCHMARK = [
    # ===== TYPE A: Single domestic =====
    {
        "id": "A01", "type": "type_a", "domain": "labor",
        "expected_source": "domestic", "expected_alignment": "no_international",
        "query": "Hợp đồng lao động phải có những nội dung gì theo quy định hiện hành?",
        "reference": "Theo Điều 21 Bộ luật Lao động 2019, hợp đồng lao động phải có các nội dung chủ yếu: tên địa chỉ người sử dụng lao động, thông tin người lao động, công việc và địa điểm làm việc, thời hạn hợp đồng, mức lương và hình thức trả lương, thời giờ làm việc và nghỉ ngơi, bảo hiểm xã hội, đào tạo bồi dưỡng."
    },
    {
        "id": "A02", "type": "type_a", "domain": "labor",
        "expected_source": "domestic", "expected_alignment": "no_international",
        "query": "Thời gian thử việc tối đa theo pháp luật lao động Việt Nam là bao lâu?",
        "reference": "Theo Bộ luật Lao động 2019, thời gian thử việc không quá 180 ngày với công việc của người quản lý doanh nghiệp, không quá 60 ngày với công việc có chức danh nghề nghiệp cần trình độ chuyên môn kỹ thuật từ cao đẳng trở lên, không quá 30 ngày với các trường hợp còn lại."
    },
    {
        "id": "A03", "type": "type_a", "domain": "labor",
        "expected_source": "domestic", "expected_alignment": "no_international",
        "query": "Người lao động có quyền đơn phương chấm dứt hợp đồng lao động không xác định thời hạn trong trường hợp nào?",
        "reference": "Theo Bộ luật Lao động 2019, người lao động làm việc theo hợp đồng không xác định thời hạn có quyền đơn phương chấm dứt hợp đồng nhưng phải báo trước ít nhất 45 ngày."
    },
    {
        "id": "A04", "type": "type_a", "domain": "labor",
        "expected_source": "domestic", "expected_alignment": "no_international",
        "query": "Mức lương tối thiểu vùng được quy định như thế nào?",
        "reference": "Lương tối thiểu vùng được Chính phủ quy định và điều chỉnh định kỳ trên cơ sở khuyến nghị của Hội đồng tiền lương quốc gia, áp dụng theo 4 vùng khác nhau tùy theo điều kiện kinh tế xã hội của từng địa phương."
    },
    {
        "id": "A05", "type": "type_a", "domain": "labor",
        "expected_source": "domestic", "expected_alignment": "no_international",
        "query": "Thời giờ làm việc bình thường của người lao động theo quy định là bao nhiêu giờ?",
        "reference": "Theo Bộ luật Lao động 2019, thời giờ làm việc bình thường không quá 8 giờ trong 1 ngày và không quá 48 giờ trong 1 tuần."
    },
    {
        "id": "A06", "type": "type_a", "domain": "food_safety",
        "expected_source": "domestic", "expected_alignment": "no_international",
        "query": "Điều kiện bảo đảm an toàn thực phẩm trong sản xuất thực phẩm là gì?",
        "reference": "Theo Luật An toàn thực phẩm, cơ sở sản xuất thực phẩm phải có địa điểm, diện tích thích hợp, có khoảng cách an toàn với nguồn ô nhiễm; có đủ nước sạch; có trang thiết bị phù hợp; đảm bảo điều kiện vệ sinh; có người quản lý đủ năng lực."
    },
    {
        "id": "A07", "type": "type_a", "domain": "food_safety",
        "expected_source": "domestic", "expected_alignment": "no_international",
        "query": "Cơ sở sản xuất thực phẩm có cần được cấp Giấy chứng nhận đủ điều kiện an toàn thực phẩm không?",
        "reference": "Theo Luật An toàn thực phẩm và Nghị định 15/2018, cơ sở sản xuất, kinh doanh thực phẩm phải có Giấy chứng nhận cơ sở đủ điều kiện an toàn thực phẩm, trừ một số trường hợp được miễn theo quy định."
    },
    {
        "id": "A08", "type": "type_a", "domain": "food_safety",
        "expected_source": "domestic", "expected_alignment": "no_international",
        "query": "Quy định về ghi nhãn thực phẩm bao gồm những nội dung gì?",
        "reference": "Nhãn thực phẩm phải ghi tên thực phẩm, thành phần, hàm lượng tịnh, ngày sản xuất, hạn sử dụng, hướng dẫn bảo quản và sử dụng, tên và địa chỉ cơ sở sản xuất, xuất xứ hàng hóa."
    },
    {
        "id": "A09", "type": "type_a", "domain": "food_safety",
        "expected_source": "domestic", "expected_alignment": "no_international",
        "query": "Thực phẩm nhập khẩu vào Việt Nam phải đáp ứng những yêu cầu gì?",
        "reference": "Thực phẩm nhập khẩu phải đáp ứng yêu cầu an toàn thực phẩm của Việt Nam và được kiểm tra nhà nước về an toàn thực phẩm tại cửa khẩu, phải có đầy đủ giấy tờ chứng minh xuất xứ và kiểm dịch."
    },
    {
        "id": "A10", "type": "type_a", "domain": "food_safety",
        "expected_source": "domestic", "expected_alignment": "no_international",
        "query": "Phụ gia thực phẩm được sử dụng trong sản xuất thực phẩm phải tuân thủ những quy định gì?",
        "reference": "Chỉ được sử dụng phụ gia thực phẩm trong danh mục được phép sử dụng do Bộ Y tế ban hành, đúng đối tượng thực phẩm, đúng liều lượng và phải ghi rõ trên nhãn."
    },

    # ===== TYPE B: Multi-hop domestic =====
    {
        "id": "B01", "type": "type_b", "domain": "labor",
        "expected_source": "domestic", "expected_alignment": "no_international",
        "query": "Nếu người sử dụng lao động đơn phương chấm dứt hợp đồng trái pháp luật thì phải bồi thường những gì cho người lao động?",
        "reference": "Theo Bộ luật Lao động 2019, người sử dụng lao động đơn phương chấm dứt hợp đồng trái pháp luật phải nhận người lao động trở lại làm việc, trả tiền lương và các quyền lợi trong thời gian không được làm việc, trả thêm ít nhất 2 tháng tiền lương. Nếu người lao động không muốn trở lại, người sử dụng phải trả thêm trợ cấp thôi việc."
    },
    {
        "id": "B02", "type": "type_b", "domain": "labor",
        "expected_source": "domestic", "expected_alignment": "no_international",
        "query": "Điều kiện để thành lập tổ chức đại diện người lao động tại doanh nghiệp là gì và tổ chức này có quyền hạn gì?",
        "reference": "Theo Bộ luật Lao động 2019 và Nghị định 145/2020, tổ chức đại diện người lao động tại doanh nghiệp được thành lập khi có đủ số lượng thành viên tối thiểu theo quy định. Tổ chức này có quyền thương lượng tập thể, đình công hợp pháp và đại diện cho người lao động trong quan hệ lao động."
    },
    {
        "id": "B03", "type": "type_b", "domain": "labor",
        "expected_source": "domestic", "expected_alignment": "no_international",
        "query": "Thủ tục giải quyết tranh chấp lao động cá nhân theo trình tự như thế nào?",
        "reference": "Theo Bộ luật Lao động 2019, tranh chấp lao động cá nhân phải qua hòa giải viên lao động trước khi khởi kiện ra Tòa án, trừ một số tranh chấp được khởi kiện thẳng ra Tòa án. Thời hạn hòa giải là 5 ngày làm việc."
    },
    {
        "id": "B04", "type": "type_b", "domain": "labor",
        "expected_source": "domestic", "expected_alignment": "no_international",
        "query": "Người lao động nước ngoài làm việc tại Việt Nam phải đáp ứng những điều kiện gì và có cần giấy phép lao động không?",
        "reference": "Theo Nghị định 152/2020, người lao động nước ngoài làm việc tại Việt Nam phải có giấy phép lao động do Bộ Lao động Thương binh và Xã hội hoặc Sở Lao động cấp, trừ một số trường hợp miễn. Phải đáp ứng điều kiện sức khỏe, trình độ chuyên môn và không có tiền án về an ninh quốc gia."
    },
    {
        "id": "B05", "type": "type_b", "domain": "labor",
        "expected_source": "domestic", "expected_alignment": "no_international",
        "query": "Chế độ trợ cấp thôi việc và trợ cấp mất việc làm khác nhau như thế nào?",
        "reference": "Theo Bộ luật Lao động 2019, trợ cấp thôi việc áp dụng khi người lao động nghỉ việc tự nguyện, mức 1/2 tháng lương cho mỗi năm làm việc. Trợ cấp mất việc làm áp dụng khi doanh nghiệp thay đổi cơ cấu, mức 1 tháng lương cho mỗi năm làm việc, tối thiểu 2 tháng lương."
    },
    {
        "id": "B06", "type": "type_b", "domain": "food_safety",
        "expected_source": "domestic", "expected_alignment": "no_international",
        "query": "Doanh nghiệp xuất khẩu thực phẩm sang EU cần phải đáp ứng những yêu cầu pháp lý nào theo quy định Việt Nam?",
        "reference": "Doanh nghiệp xuất khẩu thực phẩm sang EU cần có Giấy chứng nhận đủ điều kiện an toàn thực phẩm, đăng ký với cơ quan có thẩm quyền, đáp ứng các quy định về kiểm dịch, truy xuất nguồn gốc và ghi nhãn theo quy định của cả Việt Nam và EU."
    },
    {
        "id": "B07", "type": "type_b", "domain": "food_safety",
        "expected_source": "domestic", "expected_alignment": "no_international",
        "query": "Quy trình thu hồi sản phẩm thực phẩm không an toàn được quy định như thế nào?",
        "reference": "Theo Luật An toàn thực phẩm, khi phát hiện thực phẩm không an toàn, tổ chức cá nhân sản xuất kinh doanh phải chủ động thu hồi, thông báo cho cơ quan nhà nước có thẩm quyền và người tiêu dùng, xử lý thực phẩm thu hồi theo quy định."
    },
    {
        "id": "B08", "type": "type_b", "domain": "food_safety",
        "expected_source": "domestic", "expected_alignment": "no_international",
        "query": "Trách nhiệm của Bộ Y tế và Bộ Nông nghiệp trong quản lý an toàn thực phẩm được phân công như thế nào?",
        "reference": "Theo Luật An toàn thực phẩm và Nghị định 15/2018, Bộ Y tế quản lý thực phẩm chức năng, phụ gia, nước uống đóng gói; Bộ Nông nghiệp quản lý nông sản, thủy sản, rau quả tươi sống; Bộ Công Thương quản lý thực phẩm chế biến, đồ uống có cồn."
    },
    {
        "id": "B09", "type": "type_b", "domain": "food_safety",
        "expected_source": "domestic", "expected_alignment": "no_international",
        "query": "Mức xử phạt vi phạm hành chính về an toàn thực phẩm như thế nào?",
        "reference": "Theo Nghị định 115/2018 sửa đổi bởi Nghị định 124/2021, mức phạt tiền tối đa về an toàn thực phẩm đối với cá nhân là 100 triệu đồng, đối với tổ chức là 200 triệu đồng, ngoài ra có thể bị đình chỉ hoạt động và thu hồi giấy phép."
    },
    {
        "id": "B10", "type": "type_b", "domain": "labor",
        "expected_source": "domestic", "expected_alignment": "no_international",
        "query": "Quy định về làm thêm giờ tối đa và điều kiện để được làm thêm giờ là gì?",
        "reference": "Theo Bộ luật Lao động 2019, thời gian làm thêm không quá 40 giờ/tháng và 200 giờ/năm, một số ngành được tối đa 300 giờ/năm. Điều kiện làm thêm phải được người lao động đồng ý, trả lương tăng thêm theo quy định và đảm bảo sức khỏe người lao động."
    },

    # ===== TYPE C: Cross-corpus =====
    {
        "id": "C01", "type": "type_c", "domain": "labor",
        "expected_source": "both", "expected_alignment": "conflict",
        "query": "Việt Nam có đáp ứng các tiêu chuẩn lao động của CPTPP về tự do hiệp hội không?",
        "reference": "CPTPP Điều 19.3 yêu cầu các bên bảo đảm quyền tự do hiệp hội và thương lượng tập thể theo tiêu chuẩn ILO. Pháp luật Việt Nam hiện hành quy định cơ cấu công đoàn duy nhất dưới sự lãnh đạo của Đảng, tạo ra khoảng cách với yêu cầu quốc tế về đa công đoàn và độc lập."
    },
    {
        "id": "C02", "type": "type_c", "domain": "labor",
        "expected_source": "both", "expected_alignment": "gap",
        "query": "EVFTA yêu cầu gì về tiêu chuẩn lao động và Việt Nam đã thực hiện như thế nào?",
        "reference": "EVFTA Chương 13 yêu cầu các bên tôn trọng và thúc đẩy các tiêu chuẩn lao động cơ bản của ILO gồm tự do hiệp hội, xóa bỏ lao động cưỡng bức, lao động trẻ em và phân biệt đối xử. Việt Nam đã phê chuẩn nhiều Công ước ILO nhưng vẫn còn khoảng cách trong thực thi về tự do hiệp hội."
    },
    {
        "id": "C03", "type": "type_c", "domain": "food_safety",
        "expected_source": "both", "expected_alignment": "gap",
        "query": "Yêu cầu kiểm dịch thực vật của EVFTA đối với thực phẩm xuất khẩu từ Việt Nam sang EU là gì?",
        "reference": "EVFTA Chương 6 về SPS yêu cầu các biện pháp kiểm dịch phải dựa trên bằng chứng khoa học và tiêu chuẩn quốc tế. Việt Nam phải đảm bảo hệ thống kiểm soát an toàn thực phẩm tương đương với EU, được EU công nhận để được hưởng ưu đãi thuế quan."
    },
    {
        "id": "C04", "type": "type_c", "domain": "food_safety",
        "expected_source": "both", "expected_alignment": "gap",
        "query": "CPTPP có những quy định nào về an toàn thực phẩm và Việt Nam cần điều chỉnh pháp luật như thế nào?",
        "reference": "CPTPP Chương 7 về SPS yêu cầu áp dụng các biện pháp dựa trên đánh giá rủi ro khoa học và tiêu chuẩn Codex, IPPC, OIE. Việt Nam cần tăng cường hệ thống kiểm tra, công nhận tương đương và thủ tục minh bạch trong quản lý an toàn thực phẩm."
    },
    {
        "id": "C05", "type": "type_c", "domain": "labor",
        "expected_source": "both", "expected_alignment": "aligned",
        "query": "Quy định về cấm lao động cưỡng bức trong pháp luật Việt Nam có phù hợp với yêu cầu của CPTPP không?",
        "reference": "CPTPP Điều 19.6 yêu cầu cấm mọi hình thức lao động cưỡng bức. Pháp luật Việt Nam nghiêm cấm lao động cưỡng bức tại Bộ luật Lao động và Bộ luật Hình sự. Tuy nhiên vẫn cần tăng cường cơ chế giám sát và thực thi hiệu quả hơn."
    },
    {
        "id": "C06", "type": "type_c", "domain": "labor",
        "expected_source": "both", "expected_alignment": "aligned",
        "query": "Tiêu chuẩn về xóa bỏ lao động trẻ em trong EVFTA so với quy định Việt Nam như thế nào?",
        "reference": "EVFTA Điều 13.4 tham chiếu Công ước ILO về xóa bỏ lao động trẻ em. Pháp luật Việt Nam cấm sử dụng lao động trẻ em dưới 15 tuổi cho công việc nặng nhọc, độc hại. Việt Nam đã phê chuẩn Công ước ILO 138 và 182 về lao động trẻ em."
    },
    {
        "id": "C07", "type": "type_c", "domain": "food_safety",
        "expected_source": "both", "expected_alignment": "gap",
        "query": "Nguyên tắc tương đương trong EVFTA về an toàn thực phẩm có nghĩa là gì và tác động đến doanh nghiệp Việt Nam như thế nào?",
        "reference": "EVFTA Chương 6 quy định nguyên tắc tương đương cho phép một bên công nhận biện pháp SPS của bên kia là tương đương nếu đạt cùng mức độ bảo vệ. Doanh nghiệp Việt Nam xuất khẩu sang EU được hưởng lợi khi hệ thống kiểm soát Việt Nam được EU công nhận tương đương."
    },
    {
        "id": "C08", "type": "type_c", "domain": "food_safety",
        "expected_source": "both", "expected_alignment": "gap",
        "query": "CPTPP yêu cầu gì về minh bạch trong quản lý an toàn thực phẩm và Việt Nam đáp ứng như thế nào?",
        "reference": "CPTPP Chương 7 yêu cầu thông báo trước về biện pháp SPS mới, cho phép góp ý và giải thích cơ sở khoa học. Việt Nam đã có cơ chế tham vấn nhưng cần cải thiện về thời hạn thông báo và tính nhất quán trong áp dụng."
    },
    {
        "id": "C09", "type": "type_c", "domain": "labor",
        "expected_source": "both", "expected_alignment": "gap",
        "query": "Quy định về không phân biệt đối xử trong lao động của EVFTA so với pháp luật Việt Nam như thế nào?",
        "reference": "EVFTA Điều 13.4 tham chiếu Công ước ILO về xóa bỏ phân biệt đối xử trong việc làm. Bộ luật Lao động Việt Nam cấm phân biệt đối xử theo giới tính, dân tộc, tôn giáo. Tuy nhiên cần tăng cường cơ chế thực thi và xử lý vi phạm hiệu quả hơn."
    },
    {
        "id": "C10", "type": "type_c", "domain": "food_safety",
        "expected_source": "both", "expected_alignment": "gap",
        "query": "Hệ thống truy xuất nguồn gốc thực phẩm của Việt Nam có đáp ứng yêu cầu của EVFTA không?",
        "reference": "EVFTA yêu cầu hệ thống truy xuất nguồn gốc đáp ứng tiêu chuẩn quốc tế để đảm bảo an toàn thực phẩm. Việt Nam đã ban hành quy định về truy xuất nguồn gốc nhưng việc áp dụng chưa đồng đều, đặc biệt với doanh nghiệp nhỏ và vừa xuất khẩu sang EU."
    },
]

def build_benchmark():
    BENCHMARK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BENCHMARK_PATH, "w", encoding="utf-8") as f:
        json.dump(BENCHMARK, f, ensure_ascii=False, indent=2)

    type_counts = {}
    alignment_counts = {}
    for q in BENCHMARK:
        type_counts[q["type"]] = type_counts.get(q["type"], 0) + 1
        alignment_counts[q["expected_alignment"]] = alignment_counts.get(q["expected_alignment"], 0) + 1

    logger.success(f"Benchmark saved → {BENCHMARK_PATH}")
    logger.info(f"Total: {len(BENCHMARK)} questions")
    for t, c in type_counts.items():
        logger.info(f"  {t}: {c}")
    logger.info("Expected alignments:")
    for a, c in alignment_counts.items():
        logger.info(f"  {a}: {c}")

if __name__ == "__main__":
    build_benchmark()