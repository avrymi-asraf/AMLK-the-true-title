# Failure Examples — v3 (Qwen3-2B, whole variant)

#status/done

Curated examples of the **biggest failure modes** in the v3 evaluation run (3-epoch LoRA, anti-degeneration decode). Labels from Gemini `gemini-2.5-flash-lite` on a 100-example stratified sample (`evaluation/error_analysis.py`). Full reports: `outputs/results/finetuned-v3.errors.json`, `outputs/results/base-v3.errors.json`.

See also: [Prediction Failure Modes](obsidian/Prediction%20Failure%20Modes.md) (v1 repetition-loop analysis), [Current Results](obsidian/Current%20Results.md).

---

## Summary rates (n=100 each)

| Failure type | v3 Finetuned | v3 Base |
|---|---|---|
| hallucination | 67% | 28% |
| omission | 71% | 22% |
| entity_or_number_error | 67% | 39% |
| lead_copying | 26% | 4% |
| fluency_problem | 17% | 3% |
| wrong_language (post-hoc) | 0% | 97% |

**Finetuned:** fails on *content* (hallucination/omission) but always answers in Hebrew.
**Base:** fails on *task completion* (English thinking, no Hebrew answer) unless chat template is used.

---

## Finetuned v3

### 1. entity or number error+hallucination+omission

**Labels:** `entity_or_number_error, hallucination, omission`

**Hallucination + wrong entities + missing main points** (20% of finetuned sample). The model writes a plausible Hebrew media-digest in the right format, but invents newspapers, people, and claims; it also drops key stories from the article.

**Article (excerpt):**
```
טרם נמצאה תשובה חד-משמעית לשאלה מדוע דווקא המלחמה בגיאורגיה הרחוקה הפכה לאייטם המרכזי של התקשורת הישראלית בשבוע האחרון, אולם היום מתברר ממילא שזו היתה רק אתנחתא: הנה חוזרים הימים העליזים של פרשיות אולמרט. שערי "מעריב" ו"ידיעות אחרונות" מוקדשים היום למשה (מוריס) טלנסקי, מי שהעיד כי נתן מעטפות מלאות דולרים לאהוד אולמרט. פרקליטיו האמריקאים של טלנסקי הודיעו אתמול כי הוא לא יחזור להמשך החקירה בישראל, מחשש שהיא תסבך אותו עם השלטונות בארצות-הברית. ב"ידיעות אחרונות" הכותרת היא "הקרב על עדות טלנסקי"; ב"הארץ", הפרשה תופסת מקום משני על השער, והכותרת היא "טלנסקי הודיע שלא יבוא להשלמת עדותו בישראל". ב"מערי…
```

**Model prediction:**
```
העיתונות מגישה סיפורים על אולmרט | ובשבוע הבא – על אורי בלוך | "דה-מרקר" מבצע ניתוח לעומק | "ישראל היום" מאשים את ישראל בהשמדת הסברה | "24 שעות" מפרסמים ידיעה שקולה לגרסה של "הארט-מיסטרי" | "מערכות" מכחישה את "הגינגריץ" | ו"גלובס" מראה לנו כמה אפשר לעשות עם אלמוניות אחת כמו זו של אורי אדרנבורו#10365 - 27.9.2009 10:00 AM | "העין השביעית" מוסר כי אין בו שימוש בכלל | "ממון" מודה כי הוא היה פעיל בפעילות חברתית של "מערכת הביניים" | אורי אליאס מפרגן לתקשורתי הזאת | "גלריה" מציגה את "מדור חי" | ב"דה-מרקר" מראים את "מעלות
```

**Reference:**
```
העיתונים מלווים את הפסקת האש החלקית בגיאורגיה | ומנצלים את המקום שהתפנה כדי לחזור ולטפל בפרשת אולמרט-טלנסקי | ויולי תמיר מקצצת בסייעות לילדים מוגבלים
```

**What went wrong:**
- Prediction invents outlets, people, or events not supported by the article.
- Key stories in the reference (and article) are missing from the prediction.
- Names/numbers are wrong or garbled (e.g. mixed Hebrew/Latin characters).

---

### 2. hallucination+omission

**Labels:** `hallucination, omission`

**Hallucination + omission** (15%). The summary is on-topic in tone but describes events that are not in the article and skips the article's actual main points.

**Article (excerpt):**
```
הידיעה כי חברות תעופה זרות מאירופה ומארה"ב החליטו להפסיק את הטיסות לישראל עד להודעה חדשה, בשל איום הטילים על שדה התעופה בן-גוריון, מגיעה לכותרת הראשית של "הארץ" ומדווחת באופן בולט יותר או בולט פחות על שערי שאר העיתונים. תמונות שמונת חיילי צה"ל שנהרגו ביממה החולפת במבצע "צוק איתן" מודפסות על שערי כל העיתונים חוץ מ"מעריב". הכותרת הראשית של העיתון מוקדשת לחייל הנעדר אורון שאול: "מחפשים סימן לגורלו של החייל הנעדר". שאול היה בנגמ"ש שנפגע מטיל ויושביו נהרגו, אך בצבא התקשו לזהות בוודאות את שרידי גופתו בין גופות ששת חבריו האחרים ברכב המשוריין והוא הוכרז כנעדר, בעוד שחמאס טוען כי נחטף על-ידיו. העיתונים…
```

**Model prediction:**
```
העיתונים מדווחים על המתרחש | "יחסים" מפרסם ידיעה על פרשת רוזנטל | וב"מערכות" מכריעים לקדם את האינטרסים של נתניהو ושטרנפלד על חשבון של אהוד ברק וחברי בית-המשפט המחוזי שלו. אבל הם כבר לא יכולים לדבר על כך. זו עוד פעם שאתה מפסיד בתחקיר על 50 שנה של עיתונות? | ו"ישראל דייוויד" מודה לוויקיפדיה על הגדתה שלהם בנושא עיתון ישראلي שקיים בכל העולם. וככה זה – הכל כאן, וכולם כאן. ככה זה בעיתונאות. כפה כסף, כפה כתבות, ככה אפשר להיות עורך. ככותבים אנחנו, ככותביהם אנחנו. כותביהו אנחנו! ככה אני, ככתביה אנחנו - ככה עיתונים שלנו, כמודעות שלנו, כמו ככה גם היום. כן, זה ככה, כן זה כ
```

**Reference:**
```
פרשני הטבלואידים מתחממים | הקרב על תקציב הביטחון לא הפסיק | העיתונים הכלכליים הופכים פסימיים יותר ודשנים פחות
```

**What went wrong:**
- Prediction invents outlets, people, or events not supported by the article.
- Key stories in the reference (and article) are missing from the prediction.

---

### 3. entity or number error+omission

**Labels:** `entity_or_number_error, omission`

**Wrong entities + omission** (10%). Names/outlets are garbled or wrong; main content missing.

**Article (excerpt):**
```
לפני עשור ישבו פקידי הממשלה ונציגי משפחת הון גדולה וביקשו להגיע להבנות. על כף המאזניים עמד סכום כסף גדול מאוד, שמקורו במשאבי הטבע של הציבור. אם היו מכניסים אותו לשק, היה כבד יותר ממשקל גופם של כל הפקידים והאדונים גם יחד. האדונים ביקשו לשמור את הכסף לעצמם, והפקידים ניסו לייצג את הציבור, שכמובן לא היה נוכח בחדר הסגור. בסופו של עניין נחתם הסכם חגיגי, ודוברי הפקידים הפיצו מסרים מלאי גאווה כל כך שהנה, הכריחו את אדוני הארץ להתחייב לשלם. שנים לאחר מכן התגלה הבלוף: הם רק שכחו לכתוב תאריך על הצק. זה קרה ב-2012. מצד אחד החשבת הכללית מיכל עבאדי בויאנגו. מן הצד השני עידן עופר. הרבה דברים יש בהסכם קציר המל…
```

**Model prediction:**
```
הממשלת ישראל חתמה על הסכם קצר עם כילו בעקבות מחזורי המלחה • מדובר בחברת שתפעל לצורך עצמי - אך היא יכולה להיות מזיקה לכלל הציבוי • וגם: מי שאתם בוחרים לדמיין שהוא ממלא את תפקידו כמערכת ביטחון, הוא בכל זאת מי שאליו תושבי הארצות אמורים לסרב • ועוד על תקשורת, ממשלות וחוקים (ואולי) על חוקי המשחק שלנו • דעה שליטה | סיכום חקירה תקשורתי של "שוק התקשורת הישראלית 21": מה אפשר ללמוד מן המקרים המפורסמים של עיתונאי, מגיש, מנהל ובכיר בקבוצות השידור הציני? • פרויקט מתגלגל: "שקיפות" ושקיפות - מה בין שניהם? • והפעם: עידan עופr, מיכל בוינגo ושמחה קוטלר • פותחים שנה: "ה
```

**Reference:**
```
על הפאול ההיסטורי של החשבת לשעבר מיכל עבאדי בויאנגו אנחנו מפסידים עכשיו מיליארדים • החלטה גורלית של הממונה על הנפט לשעבר אלכס ורשבסקי, הפכה לתרומה בשמנו על סך מאות מיליונים לענקיות הגז • איך זה קורה? חברות הענק והבנקים גורמים שוב ושוב לנציגי המדינה להתפשר על הדבר הכי חשוב ברפורמה: אלמנט הזמן • אל ייאוש - יש אור בקצה המנהרה • טור בעקבות תחקיר "שקוף"
```

**What went wrong:**
- Key stories in the reference (and article) are missing from the prediction.
- Names/numbers are wrong or garbled (e.g. mixed Hebrew/Latin characters).

---

### 4. entity or number error+hallucination+lead copying

**Labels:** `entity_or_number_error, hallucination, lead_copying`

**Hallucination + entity errors + lead bias** (8%). Picks up lead-adjacent names but fabricates the rest.

**Article (excerpt):**
```
פרשת האזנות הסתר של עיתונאי "ניוז אוף דה-וורלד" עוררה את זעמם של רבים ברחבי בריטניה, במיוחד אחרי שנודע כי העיתון האזין גם לקורבנות רצח וטרור. אלא שהמטרות הנפוצות יותר של אנשי ה"ניוז אוף דה-וורלד", כך על-פי החשד, היו סלבריטאים. גם אלו לא רוו נחת מהפלישה לפרטיותם, והתקשורת הבריטית מעריכה כי ניוז-אינטרנשיונל, התאגיד של ניוז-קורפ שבבעלות רופרט מרדוק, שהפעיל את העיתון, שילם עד כה כ-2 מיליון ליש"ט בהסדרי פשרה. אולם היה מואזן שלא הסתפק בצעדים משפטיים, והפך לאחד הגיבורים המפתיעים של הפרשה, המאיימת למוטט את האימפריה התקשורתית של מרדוק בבריטניה: שחקן הקולנוע יו גרנט, חביב הצהובונים הבריטיים וכוכב קומדיו…
```

**Model prediction:**
```
יו גמר מתוך "ניו-יורקר" • "אני חשב שכולם מכירים ממני את זה - זו היא השיטה שלנו" • ראיון עם שחקנת הקולמוס המקצועי והמעורבת בתקשורות הבריטים, שסיקר את פרשת האזור הסמוי של "הניוז" של רופерт מרנדוק • פרויקט מתגלגל של "שקוף" | "הצלחה" | ו"המודיע" של "מעריב" • כל הפרטים ברשומה כאן • וגם: טור חדש של דן מרגלית! • פוסט מתוך הספרינג פאז' 21 של "sharjah.com" • מתוך ספרו "The Art of War" של יו לגרנט • פירוט הפרשות: www.bbc.co.uk/first/article/2010/08/y_g_mak_c.html • תגובה לתגובת "העין השביעית": http://www.sharjah.net/articles.php?id=459 • פניות לעיתונ
```

**Reference:**
```
פרשת האזנות הסתר בבריטניה: שחקן הקולנוע יו גרנט הקליט בסתר עיתונאי מאנשי "ניוז אוף דה-וורלד", שפרצו לטלפון שלו. ה-BBC עימת ביניהם
```

**What went wrong:**
- Prediction invents outlets, people, or events not supported by the article.
- Names/numbers are wrong or garbled (e.g. mixed Hebrew/Latin characters).
- Over-relies on lead-adjacent entities without faithful abstraction.

---

### 5. lead copying+omission

**Labels:** `lead_copying, omission`

**Lead copying + omission** (7%). Echoes opening themes or entities but fails to summarize the full article.

**Article (excerpt):**
```
"מקורות בעלי כוח מנצלים באופן קבוע את דפוסי התקשורת ואת תלותה כדי לנהל אותה ולבצע בה מניפולציה כך שתעקוב אחר אגנדה ומסגרת מיוחדת", כתבו אדוארד ס. הרמן ונועם חומסקי בהקדמתם לספר "ייצור הסכמה – הכלכלה הפוליטית של תקשורת ההמונים". 22 שנה חלפו מאז פורסם הספר לראשונה. תרגום לעברית עדיין לא יצא לאור, אך המציאות במדינת ישראל מספקת כמעט מדי יום מקרי מבחן ל"מודל הפרופגנדה" שהציעו הרמן וחומסקי. אתמול היה תורו של חומסקי עצמו להיות בצד המסוקר. דרכו אל עבר מתן הרצאה באוניברסיטת ביר-זית הגיעה לסיום מוקדם בגבול ירדן-ישראל. משרד הפנים לא התיר לאבי הבלשנות הגנרטיבית ואחד המבקרים החריפים ביותר של ארה"ב וישראל ל…
```

**Model prediction:**
```
המשבר בין נתניהו לרפורמיזם | "ידיעה אחרונה" מגבה את הזכות לכלל | "הארצות" מפגינים עיסוק בזירות נחותות | ועדת בנק ישראל מושחתת | "מערכת חדשה" מציגה את "מדריך האינטרסים" שלה | וגיא רוזן מפרסم את "הצלחה" שלו כמושלחן | "כלכליסט" מפרגן למו"ל "ישראל דיגיטל" | "7 ימים" מודה לוויקיפדיה | "גלובס" מודיע על פעילות חברת "יפה" | ו"ימות" מזהיר מ"ישובי חברה" | וכמה מהעיתונים אחרונים מפרקים את "מפלגת העבודה" | סיכום יום שלישי, 4.5.2009 | "מקצועיים" מראיינים את אורי בלאו | "תרבות" מרכז את השיח בנושא | "תחקיר" מנסה להגיע ל"יח
```

**Reference:**
```
דן מרגלית מסביר באילו נסיבות ראוי להחרים הוגה דעות | עיתונאים חוטפים מכות ונעצרים | ו"ידיעות אחרונות" מדווח בהרחבה על התפתחות קטנטונת במשפט שרה נתניהו
```

**What went wrong:**
- Key stories in the reference (and article) are missing from the prediction.
- Over-relies on lead-adjacent entities without faithful abstraction.

---

### 6. entity or number error+fluency problem+hallucination

**Labels:** `entity_or_number_error, fluency_problem, hallucination`

**Hallucination + entity errors + fluency** (7%). Fluent-looking Hebrew with garbled tokens and invented content.

**Article (excerpt):**
```
ביום שלישי השבוע, לקראת יום האשה הבינלאומי, הגישה ח"כ זהבה גלאון הצעת חוק המחמירה מאוד את האיסורים על פרסום מתן שירותי זנות בעיתונות. מכל העיתונים הגדולים בארץ, רק בשני אלה שאין להם כל צורך או תלות בפרסומי הזנות כמקור הכנסה, "ישראל היום" ו"גלובס", התייחסו לנושא. גדעון אלון, כתב "ישראל היום" בכנסת, דיווח על הצעות חוק שונות העוסקות במעמד הנשים בישראל בידיעה שפורסמה בתחתית עמ 11, ובין היתר הזכיר גם הצעת החוק של ח"כ גלאון; במוסף היומי של "גלובס" דיווחה ענת ביין-לובוביץ, בהרחבה יחסית, על הנושא. לעומת זאת, ב"הארץ", "מעריב" ו"ידיעות אחרונות" לא היה כל זכר להצעת החוק, וזאת אף שמקום נרחב ובולט הוקדש בע…
```

**Model prediction:**
```
הצעת חקיקה שתוכננה על-ידי ח"כים מגזבת והיא אינה נושאת תואר ראשון בקרב הדיווח המקצועי על המתן לתשובות בדיון בוועדת חקירת הסחר במינוי עצמו של שופט בית המשפט הראשי דני אלון • "ידיעה אחרונה" ושבועון "הארט" מפרסמות מודעה עם תצלום של אשה שעולה על גדותיה כדי לסמן את היעד שלה עבור מפרסת זנות • "מערכאות" מציגה מודעת פרסומו של אדם שלא ידוע לו כיצד ניתן לפטר אותו מטעם עבודתו • "גלי־השמאל" מפקפק בטענה כי אין לכך כל משמעות • "העין השביעית" מודה כי "המצב הופרך" אבל לא מכוון להפעיל את הלכה למעשה • "7 ימים" מוזכר פעם אחת בלבד, ושולי • "כלכליסט" ו-"י
```

**Reference:**
```
פרק נוסף במלחמה במודעות הזנות: ח"כ זהבה גלאון הגישה הצעת חוק המחמירה מאוד את האיסורים הקיימים. "ראיתי את ההתגייסות לטרפוד הצעות דומות", היא אומרת, "הן נפלו עוד לפני שהספיקו להגיע למליאת הכנסת"
```

**What went wrong:**
- Prediction invents outlets, people, or events not supported by the article.
- Names/numbers are wrong or garbled (e.g. mixed Hebrew/Latin characters).
- Surface fluency breaks down (tokenization artifacts, incoherent phrases).

---

### 7. omission

**Labels:** `omission`

**Omission only** (8%). May look reasonable but leaves out the article's central story.

**Article (excerpt):**
```
מותו של הזמר אריק איינשטיין מסוקר בעיתוני הבוקר בהיקף המעיד על מעמדו החריג בחברה בישראל. "ידיעות אחרונות", שמוביל בשנים האחרונות את הפרידות התקשורתיות מגיבורי האליטה הישנה (מדב לאוטמן וחיים חפר ועד שמוליק קראוס וענת גוב), מקדיש לאיינשטיין את כל התוכן המערכתי עד כפולת האמצע, כולל. השוואה לעיתונים האחרים מלמדת כי על אף שהסיקור של "ידיעות אחרונות" הוא הנרחב ביותר, הוא אינו היוצא מן הכלל. גם "ישראל היום", שבשנים האחרונות הוביל את סיקור מותם של אבי ראש הממשלה ואבי רעייתו, מקדיש את רוב השער וכל התוכן המערכתי בחמש הכפולות הפותחות למות איינשטיין. "מעריב", העיתון שאתמול עוד התגאה בשערו בטור החדש שהזמר …
```

**Model prediction:**
```
"יחסו של 'ידיעה אחרונה' לגיבוריו" | "האחים-שמיד" ממשיכים לקדם את זכויות היוצרים | "יכול להיות" מדגימה את השימוש בדובר צה"ל באופן פרסומי | "מערכת המשפט" מציעה סיכוי טוב יותר לכיסוי הפרצה בגוף דיסקרטורי של נתניהו | "גלובס" מחפש את הגרסה המקצועית של הדברים שכתבת אליעזר פישמן על בנימין נתניהو | "דה-מרקר" מזהיר מפני הרפתקות תקשורתית של אלון ענקינגר | "כלכליסט" מודה לדברי אורי ברלב על כך ש"היא היא זו שכינה אותנו" | ו"ימין" מפרסم כתבה על חברו של נתן פליט, אילן רוזנטל, שרק חשב על חייו האישיים בזמן שהיה עיתון?#5698 - "ימון": "האו"ם לא יוכל לסרב
```

**Reference:**
```
אריק איינשטיין מת | צחי קומה מהמר | "ידיעות אחרונות" מדברר את צה"ל | שלמה צזנה את שרה נתניהו | ו"ממון" חושב חיובי
```

**What went wrong:**
- Key stories in the reference (and article) are missing from the prediction.

---

## Base v3 (zero-shot, raw prompt — known baseline bug)

> **Note:** A fairer base baseline uses Qwen3 chat template with `enable_thinking=False`. Regen job `6a48d621` (`--pred-suffix=-v4`) was submitted 2026-07-04.

### 1. wrong_language (non-answer)

**Labels:** `(none — wrong_language post-hoc)`

**Wrong language / non-answer** (97% of base sample, post-hoc tag). Qwen3 enters an English `<think>` block and never produces a Hebrew summary. The literature failure taxonomy marks these as "clean" because the English reasoning restates the article.

**Article (excerpt):**
```
כ-11 מיליון שקלים. זה הסכום שהציבור משלם מדי חודש למפלגות לפעילותן השוטפת. וזה בנוסף לכסף שהן קיבלו על כל מסע בחירות, בסכום שעמד על כ-180 מיליון שקלים לכל סבב שכזה. כדי להמחיש כמה כסף מקבלות המפלגות מהציבור בכל רגע נתון יצרנו ממשק שבו תוכלו להתעדכן בכך בכל שניה משעות היממה.  אמנם בין שליטה של בעלי הון בח"כים לבין העלות הציבורית (המוגזמת לדעתנו) עדיפה ההוצאה הציבורית. אבל צריך לזכור, זה הכסף שמשולם למפלגות מעבר לתשלומים הקבועים של הכנסת לחברי הכנסת: שלושה יועצים לפחות, תקציב קשר עם הציבור שעומד על כ-94 אלף שקלים בשנה ומשכורת כמובן.  הבעיה בכספי מימון מפלגות היא שהם לא שקופים. אף שמדובר בכסף ציב…
```

**Model prediction:**
```
<think>
Thinking Process:

1.  **Analyze the Request:**
    *   Input: A paragraph of Hebrew text describing a political context regarding party funding from public money vs. private capital influence. It mentions specific amounts and mechanisms for transparency/accountability.
    Task: Summarize this text.
2.  Language Constraint: The output must be written in Hebrew ("Write the summary в Hebrew").
3.  Source Text Analysis (Hebrew):
    "כ-51 מ'מיליונר שקלים." -> Wait, looking closer at the first sentence provided by user input... actually it says "כ־11" which is likely a typo or OCR error for "כְּ11". Let's re-read carefully. Ah, wait, I need to check if there are typos in my reading. Actually, let me read the raw text again.
    
    Raw text starts with: "כ‑11 מתוך..." No, that doesn't make sense. Let me look at the actual prompt content again. Oh, the prompt has "כ‐11"? No, it looks like "כָּ1" maybe? Or just "כַּ1"? Wait, no, the
```

**Reference:**
```
מאז הבחירות האחרונות קיבלו המפלגות השונות כ-54 מיליון שקלים מהקופה הציבורית. ואף שמדובר בכסף ציבורי, הוא אינו שקוף • הכלי של "שקוף" יסייע לכם להכניס את הענייני לפרופורציה - למשל כמה עובדים אפשר להעסיק בחודש בכסף הזה? יש לכם בסיס להשוואה משלכם? כתבו לנו ונוסיף לכלי
```

**What went wrong:**
- Model outputs English reasoning inside `<think>` and often hits `max_new_tokens` before any Hebrew summary.
- Judge taxonomy has no "wrong language" label, so this often scores as "no failure".

---

### 2. entity or number error+hallucination

**Labels:** `entity_or_number_error, hallucination`

**Hallucination + entity errors in English** (13% of base). Responds in English with invented facts (e.g. dollar amounts not in the source).

**Article (excerpt):**
```
טרם נמצאה תשובה חד-משמעית לשאלה מדוע דווקא המלחמה בגיאורגיה הרחוקה הפכה לאייטם המרכזי של התקשורת הישראלית בשבוע האחרון, אולם היום מתברר ממילא שזו היתה רק אתנחתא: הנה חוזרים הימים העליזים של פרשיות אולמרט. שערי "מעריב" ו"ידיעות אחרונות" מוקדשים היום למשה (מוריס) טלנסקי, מי שהעיד כי נתן מעטפות מלאות דולרים לאהוד אולמרט. פרקליטיו האמריקאים של טלנסקי הודיעו אתמול כי הוא לא יחזור להמשך החקירה בישראל, מחשש שהיא תסבך אותו עם השלטונות בארצות-הברית. ב"ידיעות אחרונות" הכותרת היא "הקרב על עדות טלנסקי"; ב"הארץ", הפרשה תופסת מקום משני על השער, והכותרת היא "טלנסקי הודיע שלא יבוא להשלמת עדותו בישראל". ב"מערי…
```

**Model prediction:**
```
The article discusses the recent testimony of Moshe Telansky regarding his $10 million bribe to Ehud Olmert and why it is not being used against him yet; while some media outlets suggest that despite this evidence, he will still be prosecuted for other reasons such as hiding assets or refusing cooperation with US authorities due to a lack of legal defense from Israel's prosecutors who claim they are protecting him by withholding information about their own actions. The article also mentions that Dori Gvazi reports on the closure of the case involving Ahad Berk because she was involved in an ongoing civil dispute between them, which prevented the prosecutor from opening the file again after six years without reopening it. Additionally, there were allegations made by various political parties' activists claiming that during the election campaign, members of one party attacked others violently, but these claims have been dismissed by the police department head Yehoshua Lamberger based on internal conflicts within the party. Finally, David Regb reported on the reduction of support given to children with disabilities at schools where teachers and school principals had reduced workloads …
```

**Reference:**
```
העיתונים מלווים את הפסקת האש החלקית בגיאורגיה | ומנצלים את המקום שהתפנה כדי לחזור ולטפל בפרשת אולמרט-טלנסקי | ויולי תמיר מקצצת בסייעות לילדים מוגבלים
```

**What went wrong:**
- Prediction invents outlets, people, or events not supported by the article.
- Names/numbers are wrong or garbled (e.g. mixed Hebrew/Latin characters).
- Responds in English, not Hebrew.

---

### 3. entity or number error+hallucination+omission base

**Labels:** `entity_or_number_error, hallucination, omission`

**English summary with wrong facts and missing points** (11% of base).

**Article (excerpt):**
```
לפני עשור ישבו פקידי הממשלה ונציגי משפחת הון גדולה וביקשו להגיע להבנות. על כף המאזניים עמד סכום כסף גדול מאוד, שמקורו במשאבי הטבע של הציבור. אם היו מכניסים אותו לשק, היה כבד יותר ממשקל גופם של כל הפקידים והאדונים גם יחד. האדונים ביקשו לשמור את הכסף לעצמם, והפקידים ניסו לייצג את הציבור, שכמובן לא היה נוכח בחדר הסגור. בסופו של עניין נחתם הסכם חגיגי, ודוברי הפקידים הפיצו מסרים מלאי גאווה כל כך שהנה, הכריחו את אדוני הארץ להתחייב לשלם. שנים לאחר מכן התגלה הבלוף: הם רק שכחו לכתוב תאריך על הצק. זה קרה ב-2012. מצד אחד החשבת הכללית מיכל עבאדי בויאנגו. מן הצד השני עידן עופר. הרבה דברים יש בהסכם קציר המל…
```

**Model prediction:**
```
The text criticizes a government contract with oil giant Chevron (Chevron) for delaying and underestimating its share of the public funds from the drilling rights to the Mediterranean Sea, highlighting that despite promises made by Prime Minister Yair Lapid's administration regarding timelines and penalties for non-compliance, there was no clear deadline or penalty clause included in the agreement itself. The author argues that this lack of transparency allows officials like Alexander Vasserman to evade accountability while enriching themselves through high-profile deals without fulfilling their obligations to the state. This situation is further illustrated by the example of gas exploration projects where delays were not only common but also justified as necessary steps due to internal bureaucratic processes rather than external factors affecting profitability. Ultimately, the article suggests that political leaders prioritize personal gain over public interest when it comes to securing contracts and managing resources effectively.assistant



המאמרcriticizes את חוזה המלכה בין ממשלת ישראל לבין חברת הנפט "כיב" (CheVRon), המתארת כיחסים ביניהם ובמיוחד את הערכות המשטריות של משרדי האנר…
```

**Reference:**
```
על הפאול ההיסטורי של החשבת לשעבר מיכל עבאדי בויאנגו אנחנו מפסידים עכשיו מיליארדים • החלטה גורלית של הממונה על הנפט לשעבר אלכס ורשבסקי, הפכה לתרומה בשמנו על סך מאות מיליונים לענקיות הגז • איך זה קורה? חברות הענק והבנקים גורמים שוב ושוב לנציגי המדינה להתפשר על הדבר הכי חשוב ברפורמה: אלמנט הזמן • אל ייאוש - יש אור בקצה המנהרה • טור בעקבות תחקיר "שקוף"
```

**What went wrong:**
- Prediction invents outlets, people, or events not supported by the article.
- Key stories in the reference (and article) are missing from the prediction.
- Names/numbers are wrong or garbled (e.g. mixed Hebrew/Latin characters).
- Responds in English, not Hebrew.

---
