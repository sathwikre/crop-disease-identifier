import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
from PIL import Image

# Initialize the Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['SECRET_KEY'] = 'supersecretkey'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


# Load the model
model_path = os.path.join(os.path.dirname(__file__), 'Team3model.h5')
model = tf.keras.models.load_model(model_path, compile=False)

# Manually configure the model's loss function
model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(reduction='sum_over_batch_size'))

img_width, img_height = 256, 256

# Class labels
class_labels = ['Bell Pepper-bacterial spot', 'Bell Pepper-healthy', 'Cassava-Bacterial Blight (CBB)',
                'Cassava-Brown Streak Disease (CBSD)', 'Cassava-Green Mottle (CGM)', 'Cassava-Healthy',
                'Cassava-Mosaic Disease (CMD)', 'Corn-cercospora leaf spot gray leaf spot', 'Corn-common rust',
                'Corn-healthy', 'Corn-northern leaf blight', 'Grape-black rot', 'Grape-esca (black measles)',
                'Grape-healthy', 'Grape-leaf blight (isariopsis leaf spot)', 'Mango-Anthracnose Fungal Leaf Disease',
                'Mango-Healthy Leaf', 'Mango-Rust Leaf Disease', 'Potato-early blight', 'Potato-healthy',
                'Potato-late blight', 'Rice-BrownSpot', 'Rice-Healthy', 'Rice-Hispa', 'Rice-LeafBlast',
                'Rose-Healthy Leaf', 'Rose-Rust', 'Rose-sawfly slug', 'Tomato-bacterial spot', 'Tomato-early blight',
                'Tomato-healthy', 'Tomato-late blight', 'Tomato-leaf mold', 'Tomato-mosaic virus',
                'Tomato-septoria leaf spot', 'Tomato-spider mites two-spotted spider mite', 'Tomato-target spot',
                'Tomato-yellow leaf curl virus']


# Precaution / treatment lookup - Bilingual (English and Telugu)
def get_precaution(label: str, lang: str = 'en') -> str:
    """Return a recommended precaution or treatment string for a predicted class label."""
    # Specific exact-label mappings - English
    precaution_map_en = {
        'Bell Pepper-bacterial spot': "Fertilizer: Avoid high Nitrogen. Apply Potassium Sulfate to strengthen leaf tissues.<br><br>Tip: Spray a Calcium Nitrate solution to prevent the plant from becoming stressed, which makes it more susceptible to spots.",
        'Bell Pepper-healthy': "Use Compost or Manure to maintain organic matter, which supports the beneficial soil microbes that fight pathogens naturally.",
        'Cassava-Bacterial Blight (CBB)': "Fertilizer: Apply Muriate of Potash (MOP). High potassium levels are proven to reduce the severity of Cassava diseases.<br><br>Tip: Ensure the soil is well-drained; waterlogged soil weakens Cassava roots and invites bacterial rot.",
        'Cassava-Brown Streak Disease (CBSD)': "Fertilizer: Apply Muriate of Potash (MOP). High potassium levels help reduce disease severity.<br><br>Tip: Ensure the soil is well-drained; waterlogged soil weakens Cassava roots. Control whitefly populations and use certified virus-free cuttings.",
        'Cassava-Green Mottle (CGM)': "Control green mite infestations using biological controls or miticides. Use resistant cultivars. Ensure well-drained soil.",
        'Cassava-Healthy': "Use Compost or Manure to maintain organic matter, which supports the beneficial soil microbes that fight pathogens naturally.",
        'Cassava-Mosaic Disease (CMD)': "Fertilizer: Apply Muriate of Potash (MOP). High potassium levels are proven to reduce the severity of Cassava Mosaic Disease.<br><br>Tip: Ensure the soil is well-drained; waterlogged soil weakens Cassava roots. Control whitefly populations and use virus-free cuttings.",
        'Corn-cercospora leaf spot gray leaf spot': "Fertilizer: Use a Top-dressing of Nitrogen (Urea) only during the early growth stages. For diseased plants, apply Zinc and Magnesium foliar sprays.<br><br>Tip: Rotate corn with legumes (beans/peas) next season to break the fungal life cycle in the soil.",
        'Corn-common rust': "Fertilizer: Use a Top-dressing of Nitrogen (Urea) only during the early growth stages. For diseased plants, apply Zinc and Magnesium foliar sprays.<br><br>Tip: Rotate corn with legumes (beans/peas) next season to break the fungal life cycle in the soil.",
        'Corn-healthy': "Maintain current fertilization and watering schedule. Continue regular field monitoring.",
        'Corn-northern leaf blight': "Fertilizer: Use a Top-dressing of Nitrogen (Urea) only during the early growth stages. For diseased plants, apply Zinc and Magnesium foliar sprays.<br><br>Tip: Rotate corn with legumes (beans/peas) next season to break the fungal life cycle in the soil.",
        'Grape-black rot': "Fertilizer: Apply Boron and Copper micronutrients. Copper isn't just a fertilizer; it acts as a fungicide.<br><br>Tip: Prune the inner branches of the vine to allow sunlight and wind to dry the leaves quickly after rain.",
        'Grape-esca (black measles)': "Fertilizer: Focus on soil conditioners like Humic Acid to improve root health.<br><br>Tip: Avoid heavy pruning during wet weather as the spores enter through open wounds.",
        'Grape-healthy': "Maintain proper vine spacing and continue routine monitoring. Ensure adequate air circulation.",
        'Grape-leaf blight (isariopsis leaf spot)': "Fertilizer: Apply Boron and Copper micronutrients.<br><br>Tip: Prune the inner branches of the vine to allow sunlight and wind to dry the leaves quickly after rain.",
        'Mango-Anthracnose Fungal Leaf Disease': "Fertilizer: Apply Boron and Copper micronutrients. Copper isn't just a fertilizer; it acts as a fungicide.<br><br>Tip: Prune the inner branches of the tree to allow sunlight and wind to dry the leaves quickly after rain.",
        'Mango-Healthy Leaf': "Continue routine scouting and balanced fertilization. Maintain good orchard hygiene.",
        'Mango-Rust Leaf Disease': "Fertilizer: Apply Boron and Copper micronutrients. Copper isn't just a fertilizer; it acts as a fungicide.<br><br>Tip: Prune the inner branches of the tree to allow sunlight and wind to dry the leaves quickly after rain.",
        'Potato-early blight': "Fertilizer: Use Phosphorus-rich fertilizers (like Bone Meal) to encourage strong tuber development even if the leaves are struggling.<br><br>Tip: Harvest during dry weather and ensure the tubers are 'cured' (skin hardened) before storage to prevent rot.",
        'Potato-healthy': "Continue routine scouting and balanced fertilization. Monitor for early disease signs.",
        'Potato-late blight': "Fertilizer: Use Phosphorus-rich fertilizers (like Bone Meal) to encourage strong tuber development even if the leaves are struggling.<br><br>Tip: Harvest during dry weather and ensure the tubers are 'cured' (skin hardened) before storage to prevent rot. This is highly destructive - act quickly.",
        'Rice-BrownSpot': "Fertilizer: This disease is an 'indicator' of poor soil. Apply Potassium (K) and Manganese. Brown spot rarely occurs in well-nourished soil.",
        'Rice-Healthy': "Maintain proper water levels and continue balanced fertilization. Monitor for early disease signs.",
        'Rice-Hispa': "Manual removal of the damaged leaf tips where larvae are present. Use neem-based pesticides or recommended insecticides.",
        'Rice-LeafBlast': "Fertilizer: STOP Nitrogen application immediately if you see Blast. Excess nitrogen makes the rice plant 'succulent' and very easy for the fungus to eat.<br><br>Tip: Maintain a consistent water level in the paddy to reduce plant stress.",
        'Rose-Healthy Leaf': "Continue routine scouting and balanced fertilization. Maintain good garden hygiene.",
        'Rose-Rust': "Fertilizer: Use a slow-release Rose food high in Potassium.<br><br>Tip: For Sawflies, use Neem Oil—it acts as both a pesticide and a leaf shine that prevents rust spores from sticking.",
        'Rose-sawfly slug': "Fertilizer: Use a slow-release Rose food high in Potassium.<br><br>Tip: Use Neem Oil—it acts as both a pesticide and a leaf shine that prevents rust spores from sticking.",
        'Tomato-bacterial spot': "Fertilizer: Avoid high Nitrogen. Apply Potassium Sulfate to strengthen leaf tissues.<br><br>Tip: Spray a Calcium Nitrate solution to prevent the plant from becoming stressed, which makes it more susceptible to spots.",
        'Tomato-early blight': "Fertilizer: Use a balanced NPK 10-10-10 but supplement with Silica. Silica creates a physical barrier on the leaf that fungi cannot easily penetrate.<br><br>Tip: Mulch around the base to prevent soil-borne spores from splashing onto the leaves during watering.",
        'Tomato-healthy': "Maintain current fertilization and watering schedule. Continue regular monitoring.",
        'Tomato-late blight': "Fertilizer: Use a balanced NPK 10-10-10 but supplement with Silica. Silica creates a physical barrier on the leaf that fungi cannot easily penetrate.<br><br>Tip: Mulch around the base to prevent soil-borne spores from splashing onto the leaves during watering. This spreads rapidly - act quickly.",
        'Tomato-leaf mold': "Fertilizer: Use a balanced NPK 10-10-10 but supplement with Silica. Silica creates a physical barrier on the leaf that fungi cannot easily penetrate.<br><br>Tip: Mulch around the base. Increase ventilation and keep humidity below 85%.",
        'Tomato-mosaic virus': "Fertilizer: There is no cure for the virus, but Seaweed Extract can help the plant tolerate the stress.<br><br>Tip: Immediately remove infected plants to save the rest of the crop. Control whiteflies (the carriers).",
        'Tomato-septoria leaf spot': "Fertilizer: Avoid high Nitrogen. Apply Potassium Sulfate to strengthen leaf tissues.<br><br>Tip: Spray a Calcium Nitrate solution to prevent the plant from becoming stressed, which makes it more susceptible to spots.",
        'Tomato-spider mites two-spotted spider mite': "Increase humidity (they prefer dry heat) and use miticides or predatory mites. Ensure adequate watering.",
        'Tomato-target spot': "Fertilizer: Avoid high Nitrogen. Apply Potassium Sulfate to strengthen leaf tissues.<br><br>Tip: Spray a Calcium Nitrate solution to prevent the plant from becoming stressed, which makes it more susceptible to spots.",
        'Tomato-yellow leaf curl virus': "Fertilizer: There is no cure for the virus, but Seaweed Extract can help the plant tolerate the stress.<br><br>Tip: Immediately remove infected plants to save the rest of the crop. Control whiteflies (the carriers)."
    }

    # Telugu translations
    precaution_map_te = {
        'Bell Pepper-bacterial spot': "సారం: అధిక నైట్రోజన్ ని నివారించండి. ఆకు కణజాలాలను బలపరచడానికి పొటాషియం సల్ఫేట్ వర్తింపజేయండి.<br><br>సూచన: మొక్కను ఒత్తిడికి గురికాకుండా నిరోధించడానికి కాల్షియం నైట్రేట్ ద్రావణాన్ని స్ప్రే చేయండి, ఇది దానిని చుక్కలకు మరింత అవకాశం కల్పిస్తుంది.",
        'Bell Pepper-healthy': "కంపోస్ట్ లేదా ఎరువును ఉపయోగించండి, ఇది సేంద్రీయ పదార్థాన్ని నిర్వహిస్తుంది, ఇది రోగకారకాలతో సహజంగా పోరాడే ప్రయోజనకరమైన నేల సూక్ష్మజీవులకు మద్దతు ఇస్తుంది.",
        'Cassava-Bacterial Blight (CBB)': "సారం: పొటాష్ మ్యూరియేట్ (MOP) వర్తింపజేయండి. అధిక పొటాషియం స్థాయిలు కసావా వ్యాధుల తీవ్రతను తగ్గించడానికి నిరూపించబడ్డాయి.<br><br>సూచన: నేల బాగా నీరు పారేలా ఉండేలా చూసుకోండి; నీటితో నిండిన నేల కసావా వేర్లను బలహీనపరుస్తుంది మరియు బాక్టీరియా కుళ్ళుకు ఆహ్వానిస్తుంది.",
        'Cassava-Brown Streak Disease (CBSD)': "సారం: పొటాష్ మ్యూరియేట్ (MOP) వర్తింపజేయండి. అధిక పొటాషియం స్థాయిలు వ్యాధి తీవ్రతను తగ్గించడంలో సహాయపడతాయి.<br><br>సూచన: నేల బాగా నీరు పారేలా ఉండేలా చూసుకోండి; నీటితో నిండిన నేల కసావా వేర్లను బలహీనపరుస్తుంది. తెల్లని ఈగ జనాభాను నియంత్రించండి మరియు ధృవీకరించబడిన వైరస్-ఉచిత కటింగ్‌లను ఉపయోగించండి.",
        'Cassava-Green Mottle (CGM)': "జీవ నియంత్రణలు లేదా మైటిసైడ్‌లను ఉపయోగించి ఆకుపచ్చ మైట్ సంక్రమణలను నియంత్రించండి. నిరోధక జాతులను ఉపయోగించండి. నేల బాగా నీరు పారేలా ఉండేలా చూసుకోండి.",
        'Cassava-Healthy': "కంపోస్ట్ లేదా ఎరువును ఉపయోగించండి, ఇది సేంద్రీయ పదార్థాన్ని నిర్వహిస్తుంది, ఇది రోగకారకాలతో సహజంగా పోరాడే ప్రయోజనకరమైన నేల సూక్ష్మజీవులకు మద్దతు ఇస్తుంది.",
        'Cassava-Mosaic Disease (CMD)': "సారం: పొటాష్ మ్యూరియేట్ (MOP) వర్తింపజేయండి. అధిక పొటాషియం స్థాయిలు కసావా మొజైక్ వ్యాధి తీవ్రతను తగ్గించడానికి నిరూపించబడ్డాయి.<br><br>సూచన: నేల బాగా నీరు పారేలా ఉండేలా చూసుకోండి; నీటితో నిండిన నేల కసావా వేర్లను బలహీనపరుస్తుంది. తెల్లని ఈగ జనాభాను నియంత్రించండి మరియు వైరస్-ఉచిత కటింగ్‌లను ఉపయోగించండి.",
        'Corn-cercospora leaf spot gray leaf spot': "సారం: ప్రారంభ వృద్ధి దశలలో మాత్రమే నైట్రోజన్ (యూరియా) టాప్-డ్రెసింగ్‌ను ఉపయోగించండి. వ్యాధి గ్రస్త మొక్కలకు, జింక్ మరియు మెగ్నీషియం ఆకు స్ప్రేలను వర్తింపజేయండి.<br><br>సూచన: నేలలోని ఫంగల్ జీవిత చక్రాన్ని విచ్ఛిన్నం చేయడానికి తదుపరి సీజన్‌లో కార్న్‌ను బీన్స్/పీస్ వంటి పప్పుధాన్యాలతో తిరగండి.",
        'Corn-common rust': "సారం: ప్రారంభ వృద్ధి దశలలో మాత్రమే నైట్రోజన్ (యూరియా) టాప్-డ్రెసింగ్‌ను ఉపయోగించండి. వ్యాధి గ్రస్త మొక్కలకు, జింక్ మరియు మెగ్నీషియం ఆకు స్ప్రేలను వర్తింపజేయండి.<br><br>సూచన: నేలలోని ఫంగల్ జీవిత చక్రాన్ని విచ్ఛిన్నం చేయడానికి తదుపరి సీజన్‌లో కార్న్‌ను బీన్స్/పీస్ వంటి పప్పుధాన్యాలతో తిరగండి.",
        'Corn-healthy': "ప్రస్తుత ఎరువు మరియు నీటిపోయే షెడ్యూల్‌ను నిర్వహించండి. క్రమం తప్పకుండా ఫీల్డ్ మానిటరింగ్‌ను కొనసాగించండి.",
        'Corn-northern leaf blight': "సారం: ప్రారంభ వృద్ధి దశలలో మాత్రమే నైట్రోజన్ (యూరియా) టాప్-డ్రెసింగ్‌ను ఉపయోగించండి. వ్యాధి గ్రస్త మొక్కలకు, జింక్ మరియు మెగ్నీషియం ఆకు స్ప్రేలను వర్తింపజేయండి.<br><br>సూచన: నేలలోని ఫంగల్ జీవిత చక్రాన్ని విచ్ఛిన్నం చేయడానికి తదుపరి సీజన్‌లో కార్న్‌ను బీన్స్/పీస్ వంటి పప్పుధాన్యాలతో తిరగండి.",
        'Grape-black rot': "సారం: బోరాన్ మరియు కాపర్ సూక్ష్మ పోషకాలను వర్తింపజేయండి. కాపర్ కేవలం ఎరువు మాత్రమే కాదు; ఇది ఫంగిసైడ్‌గా పనిచేస్తుంది.<br><br>సూచన: వర్షం తర్వాత ఆకులను త్వరగా ఎండబెట్టడానికి సూర్యరశ్మి మరియు గాలిని అనుమతించడానికి తీగ యొక్క లోపలి కొమ్మలను కత్తిరించండి.",
        'Grape-esca (black measles)': "సారం: వేరు ఆరోగ్యాన్ని మెరుగుపరచడానికి హ్యూమిక్ యాసిడ్ వంటి నేల కండిషనర్‌లపై దృష్టి పెట్టండి.<br><br>సూచన: తడి వాతావరణంలో భారీ కత్తిరింపును నివారించండి ఎందుకంటే బీజాంశాలు తెరిచిన గాయాల ద్వారా ప్రవేశిస్తాయి.",
        'Grape-healthy': "సరైన తీగ అంతరాన్ని నిర్వహించండి మరియు క్రమం తప్పకుండా మానిటరింగ్‌ను కొనసాగించండి. తగిన గాలి ప్రసరణను నిర్ధారించండి.",
        'Grape-leaf blight (isariopsis leaf spot)': "సారం: బోరాన్ మరియు కాపర్ సూక్ష్మ పోషకాలను వర్తింపజేయండి.<br><br>సూచన: వర్షం తర్వాత ఆకులను త్వరగా ఎండబెట్టడానికి సూర్యరశ్మి మరియు గాలిని అనుమతించడానికి తీగ యొక్క లోపలి కొమ్మలను కత్తిరించండి.",
        'Mango-Anthracnose Fungal Leaf Disease': "సారం: బోరాన్ మరియు కాపర్ సూక్ష్మ పోషకాలను వర్తింపజేయండి. కాపర్ కేవలం ఎరువు మాత్రమే కాదు; ఇది ఫంగిసైడ్‌గా పనిచేస్తుంది.<br><br>సూచన: వర్షం తర్వాత ఆకులను త్వరగా ఎండబెట్టడానికి సూర్యరశ్మి మరియు గాలిని అనుమతించడానికి చెట్టు యొక్క లోపలి కొమ్మలను కత్తిరించండి.",
        'Mango-Healthy Leaf': "క్రమం తప్పకుండా స్కౌటింగ్ మరియు సమతుల్య ఎరువును కొనసాగించండి. మంచి తోట పరిశుభ్రతను నిర్వహించండి.",
        'Mango-Rust Leaf Disease': "సారం: బోరాన్ మరియు కాపర్ సూక్ష్మ పోషకాలను వర్తింపజేయండి. కాపర్ కేవలం ఎరువు మాత్రమే కాదు; ఇది ఫంగిసైడ్‌గా పనిచేస్తుంది.<br><br>సూచన: వర్షం తర్వాత ఆకులను త్వరగా ఎండబెట్టడానికి సూర్యరశ్మి మరియు గాలిని అనుమతించడానికి చెట్టు యొక్క లోపలి కొమ్మలను కత్తిరించండి.",
        'Potato-early blight': "సారం: బలమైన కంద రూపాంతరాన్ని ప్రోత్సహించడానికి ఫాస్ఫరస్-సమృద్ధ ఎరువులు (బోన్ మీల్ వంటివి) ఉపయోగించండి, ఆకులు కష్టపడుతున్నప్పటికీ.<br><br>సూచన: కుళ్ళుకు నిరోధకత కోసం నిల్వ చేసే ముందు పొడి వాతావరణంలో పంటను కోయండి మరియు కందలు 'క్యూర్' చేయబడ్డాయి (చర్మం గట్టిపడింది) అని నిర్ధారించండి.",
        'Potato-healthy': "క్రమం తప్పకుండా స్కౌటింగ్ మరియు సమతుల్య ఎరువును కొనసాగించండి. ప్రారంభ వ్యాధి సంకేతాల కోసం పర్యవేక్షించండి.",
        'Potato-late blight': "సారం: బలమైన కంద రూపాంతరాన్ని ప్రోత్సహించడానికి ఫాస్ఫరస్-సమృద్ధ ఎరువులు (బోన్ మీల్ వంటివి) ఉపయోగించండి, ఆకులు కష్టపడుతున్నప్పటికీ.<br><br>సూచన: కుళ్ళుకు నిరోధకత కోసం నిల్వ చేసే ముందు పొడి వాతావరణంలో పంటను కోయండి మరియు కందలు 'క్యూర్' చేయబడ్డాయి (చర్మం గట్టిపడింది) అని నిర్ధారించండి. ఇది చాలా విధ్వంసకరమైనది - త్వరగా వ్యవహరించండి.",
        'Rice-BrownSpot': "సారం: ఈ వ్యాధి పేద నేలకు 'సూచిక'. పొటాషియం (K) మరియు మాంగనీస్ వర్తింపజేయండి. బ్రౌన్ స్పాట్ బాగా పోషించబడిన నేలలో అరుదుగా సంభవిస్తుంది.",
        'Rice-Healthy': "సరైన నీటి స్థాయిలను నిర్వహించండి మరియు సమతుల్య ఎరువును కొనసాగించండి. ప్రారంభ వ్యాధి సంకేతాల కోసం పర్యవేక్షించండి.",
        'Rice-Hispa': "లార్వా ఉన్న దెబ్బతిన్న ఆకు చివరలను మాన్యువల్‌గా తీసివేయండి. నీం-ఆధారిత కీటకనాశకాలు లేదా సిఫార్సు చేసిన కీటకనాశకాలను ఉపయోగించండి.",
        'Rice-LeafBlast': "సారం: మీరు బ్లాస్ట్‌ను చూస్తే వెంటనే నైట్రోజన్ అప్లికేషన్‌ను ఆపండి. అధిక నైట్రోజన్ వరి మొక్కను 'రసవంతమైనదిగా' చేస్తుంది మరియు ఫంగస్ తినడానికి చాలా సులభం.<br><br>సూచన: మొక్క ఒత్తిడిని తగ్గించడానికి పాడీలో స్థిరమైన నీటి స్థాయిని నిర్వహించండి.",
        'Rose-Healthy Leaf': "క్రమం తప్పకుండా స్కౌటింగ్ మరియు సమతుల్య ఎరువును కొనసాగించండి. మంచి తోట పరిశుభ్రతను నిర్వహించండి.",
        'Rose-Rust': "సారం: పొటాషియంలో అధికంగా ఉన్న నెమ్మదిగా విడుదలయ్యే రోజ్ ఆహారాన్ని ఉపయోగించండి.<br><br>సూచన: సాఫ్లైస్ కోసం, నీం ఆయిల్‌ను ఉపయోగించండి—ఇది కీటకనాశకం మరియు ఆకు మెరుపు రెండింటిగా పనిచేస్తుంది, ఇది తుప్పు బీజాంశాలు అంటుకోకుండా నిరోధిస్తుంది.",
        'Rose-sawfly slug': "సారం: పొటాషియంలో అధికంగా ఉన్న నెమ్మదిగా విడుదలయ్యే రోజ్ ఆహారాన్ని ఉపయోగించండి.<br><br>సూచన: నీం ఆయిల్‌ను ఉపయోగించండి—ఇది కీటకనాశకం మరియు ఆకు మెరుపు రెండింటిగా పనిచేస్తుంది, ఇది తుప్పు బీజాంశాలు అంటుకోకుండా నిరోధిస్తుంది.",
        'Tomato-bacterial spot': "సారం: అధిక నైట్రోజన్ ని నివారించండి. ఆకు కణజాలాలను బలపరచడానికి పొటాషియం సల్ఫేట్ వర్తింపజేయండి.<br><br>సూచన: మొక్కను ఒత్తిడికి గురికాకుండా నిరోధించడానికి కాల్షియం నైట్రేట్ ద్రావణాన్ని స్ప్రే చేయండి, ఇది దానిని చుక్కలకు మరింత అవకాశం కల్పిస్తుంది.",
        'Tomato-early blight': "సారం: సమతుల్య NPK 10-10-10 ని ఉపయోగించండి కానీ సిలికాను అదనంగా జోడించండి. సిలికా ఆకుపై భౌతిక అవరోధాన్ని సృష్టిస్తుంది, ఫంగై సులభంగా చొచ్చుకురాకూడదు.<br><br>సూచన: నీటిపోయే సమయంలో నేల-జన్య బీజాంశాలు ఆకులపై చిమ్మకుండా నిరోధించడానికి బేస్ చుట్టూ మల్చ్ చేయండి.",
        'Tomato-healthy': "ప్రస్తుత ఎరువు మరియు నీటిపోయే షెడ్యూల్‌ను నిర్వహించండి. క్రమం తప్పకుండా పర్యవేక్షణను కొనసాగించండి.",
        'Tomato-late blight': "సారం: సమతుల్య NPK 10-10-10 ని ఉపయోగించండి కానీ సిలికాను అదనంగా జోడించండి. సిలికా ఆకుపై భౌతిక అవరోధాన్ని సృష్టిస్తుంది, ఫంగై సులభంగా చొచ్చుకురాకూడదు.<br><br>సూచన: నీటిపోయే సమయంలో నేల-జన్య బీజాంశాలు ఆకులపై చిమ్మకుండా నిరోధించడానికి బేస్ చుట్టూ మల్చ్ చేయండి. ఇది వేగంగా వ్యాపిస్తుంది - త్వరగా వ్యవహరించండి.",
        'Tomato-leaf mold': "సారం: సమతుల్య NPK 10-10-10 ని ఉపయోగించండి కానీ సిలికాను అదనంగా జోడించండి. సిలికా ఆకుపై భౌతిక అవరోధాన్ని సృష్టిస్తుంది, ఫంగై సులభంగా చొచ్చుకురాకూడదు.<br><br>సూచన: బేస్ చుట్టూ మల్చ్ చేయండి. గాలి ప్రసరణను పెంచండి మరియు తేమను 85% కంటే తక్కువగా ఉంచండి.",
        'Tomato-mosaic virus': "సారం: వైరస్‌కు నివారణ లేదు, కానీ సీవీడ్ ఎక్స్‌ట్రాక్ట్ మొక్క ఒత్తిడిని తట్టుకోవడంలో సహాయపడుతుంది.<br><br>సూచన: మిగిలిన పంటను రక్షించడానికి వెంటనే సంక్రమిత మొక్కలను తీసివేయండి. తెల్లని ఈగలను (వాహకాలు) నియంత్రించండి.",
        'Tomato-septoria leaf spot': "సారం: అధిక నైట్రోజన్ ని నివారించండి. ఆకు కణజాలాలను బలపరచడానికి పొటాషియం సల్ఫేట్ వర్తింపజేయండి.<br><br>సూచన: మొక్కను ఒత్తిడికి గురికాకుండా నిరోధించడానికి కాల్షియం నైట్రేట్ ద్రావణాన్ని స్ప్రే చేయండి, ఇది దానిని చుక్కలకు మరింత అవకాశం కల్పిస్తుంది.",
        'Tomato-spider mites two-spotted spider mite': "తేమను పెంచండి (వారు పొడి వేడిని ఇష్టపడతారు) మరియు మైటిసైడ్‌లు లేదా శికారి మైట్‌లను ఉపయోగించండి. తగిన నీటిపోయేలా నిర్ధారించండి.",
        'Tomato-target spot': "సారం: అధిక నైట్రోజన్ ని నివారించండి. ఆకు కణజాలాలను బలపరచడానికి పొటాషియం సల్ఫేట్ వర్తింపజేయండి.<br><br>సూచన: మొక్కను ఒత్తిడికి గురికాకుండా నిరోధించడానికి కాల్షియం నైట్రేట్ ద్రావణాన్ని స్ప్రే చేయండి, ఇది దానిని చుక్కలకు మరింత అవకాశం కల్పిస్తుంది.",
        'Tomato-yellow leaf curl virus': "సారం: వైరస్‌కు నివారణ లేదు, కానీ సీవీడ్ ఎక్స్‌ట్రాక్ట్ మొక్క ఒత్తిడిని తట్టుకోవడంలో సహాయపడుతుంది.<br><br>సూచన: మిగిలిన పంటను రక్షించడానికి వెంటనే సంక్రమిత మొక్కలను తీసివేయండి. తెల్లని ఈగలను (వాహకాలు) నియంత్రించండి."
    }

    # Select the appropriate map based on language
    precaution_map = precaution_map_en if lang == 'en' else precaution_map_te

    # Normalize label for simple pattern matches
    lower = label.lower()

    # General rules before default map lookup
    if lang == 'en':
        if 'bacterial blight' in lower or 'bacterial blight (cbb)' in lower:
            return 'Clean cuttings and tool sterilization; use resistant varieties if available.'
        if 'late blight' in lower:
            return 'Preventative fungicides and removing volunteer plants; use certified disease-free seed/tubers.'
        if 'bacterial spot' in lower and 'pepper' in lower:
            return 'Use pathogen-free seeds and avoid overhead irrigation; consider copper-based bactericides.'
        if 'healthy' in lower:
            return 'Continue routine scouting and balanced fertilization.'
        default_msg = 'No specific precaution found. Monitor and follow good cultural practices.'
    else:  # Telugu
        if 'bacterial blight' in lower or 'bacterial blight (cbb)' in lower:
            return 'కటింగ్‌లను శుభ్రం చేయడం మరియు సాధనాల స్టెరిలైజేషన్; అందుబాటులో ఉంటే నిరోధక జాతులను ఉపయోగించండి.'
        if 'late blight' in lower:
            return 'నివారక ఫంగిసైడ్‌లు మరియు స్వచ్ఛంద మొక్కలను తీసివేయడం; ధృవీకరించబడిన వ్యాధి-ఉచిత విత్తనం/కందలను ఉపయోగించండి.'
        if 'bacterial spot' in lower and 'pepper' in lower:
            return 'రోగకారక-ఉచిత విత్తనాలను ఉపయోగించండి మరియు ఓవర్‌హెడ్ నీటిపోయేలా నివారించండి; కాపర్-ఆధారిత బాక్టీరిసైడ్‌లను పరిగణించండి.'
        if 'healthy' in lower:
            return 'క్రమం తప్పకుండా స్కౌటింగ్ మరియు సమతుల్య ఎరువును కొనసాగించండి.'
        default_msg = 'నిర్దిష్ట జాగ్రత్త కనుగొనబడలేదు. పర్యవేక్షించండి మరియు మంచి సాంస్కృతిక పద్ధతులను అనుసరించండి.'

    # Exact label lookup
    return precaution_map.get(label, default_msg)

# Function to predict the class of the plant disease
def model_prediction(test_image_path):
    # Ensure image is RGB (drop alpha channel if present) and resized
    try:
        image = Image.open(test_image_path).convert('RGB')
        image = image.resize((img_width, img_height))
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = np.expand_dims(input_arr, axis=0)
        input_arr = input_arr / 255.0
        predictions = model.predict(input_arr, verbose=0)
        result_index = int(np.argmax(predictions))
        print(f"Model prediction shape: {predictions.shape}, max index: {result_index}, max value: {np.max(predictions)}")
        return result_index
    except Exception as e:
        print(f"Error in model_prediction: {str(e)}")
        raise

@app.route('/')
def index():
    # Serve the home page as the default root (login disabled)
    return render_template('disease-recognition.html')

# Login disabled: login route removed

# Removed separate /home route — root now serves the disease recognition page.

@app.route('/disease-recognition', methods=['GET', 'POST'])
def disease_recognition():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            try:
                file.save(filepath)
            except UnicodeEncodeError:
                flash('File name contains unsupported characters.')
                return redirect(request.url)
            try:
                result_index = model_prediction(filepath)
                print(f"Prediction result index: {result_index}")  # Debug output
            except Exception as e:
                print(f"Prediction error: {str(e)}")  # Debug output
                flash('Prediction error: {}'.format(str(e)))
                return redirect(request.url)
            
            # Validate result_index
            if result_index is None or result_index < 0 or result_index >= len(class_labels):
                error_msg = f'Invalid prediction index: {result_index}'
                print(error_msg)  # Debug output
                flash(error_msg)
                return redirect(request.url)
            
            prediction = class_labels[result_index]
            print(f"Predicted disease: {prediction}")  # Debug output
            
            # Get language from request (cookie or default to 'en')
            lang = request.cookies.get('language', 'en')
            if lang not in ['en', 'te']:
                lang = 'en'
            precaution = get_precaution(prediction, lang)
            print(f"Precaution retrieved: {precaution[:50]}...")  # Debug output
            
            return render_template('prediction.html', predicted_disease=prediction, precaution=precaution, image_url=url_for('static', filename='uploads/' + filename))
    return render_template('disease-recognition.html')

# Login/logout removed: no session management in this app

if __name__ == '__main__':
    app.run(debug=True)