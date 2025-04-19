import React, { useState, useRef, useEffect } from 'react';
import useDatasetStats from '../hooks/useDatasetStats';
import { InformationCircleIcon, ChartBarIcon, ArrowDownTrayIcon, LanguageIcon, AcademicCapIcon } from '@heroicons/react/24/outline';

const TechTooltip = ({ isHebrew, content, onMouseEnter, onMouseLeave, onLanguageToggle, type }) => (
  <div 
    onMouseEnter={onMouseEnter}
    onMouseLeave={onMouseLeave}
    className={`
      absolute bottom-full right-[-20px] mb-2 bg-white rounded-lg shadow-xl border border-gray-200 z-50
      ${type === 'training' ? 'w-[350px]' : 'w-[450px]'}
    `}
    dir={isHebrew ? "rtl" : "ltr"}
    style={{ 
      transform: 'translateX(0)',
      maxWidth: 'calc(100vw - 40px)'  // Prevent edge cutoff
    }}
  >
    <div className="p-6">
      <div className="flex justify-between items-start mb-4">
        <h4 className="text-lg font-medium text-gray-700">{isHebrew ? "פרטים טכניים" : "Technical Details"}</h4>
        <button 
          onClick={onLanguageToggle}
          className="text-blue-500 hover:text-blue-700"
        >
          <LanguageIcon className="w-5 h-5" />
        </button>
      </div>
      <div className="space-y-5 text-sm">
        {content}
      </div>
    </div>
    <div className="absolute bottom-0 right-[25px] transform translate-y-1/2 rotate-45 w-3 h-3 bg-white border-r border-b border-gray-200"></div>
  </div>
);

const InfoPanel = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [showTooltip, setShowTooltip] = useState(false);
  const [activeTooltip, setActiveTooltip] = useState(null);
  const [isHebrew, setIsHebrew] = useState(false);
  const tooltipRef = useRef(null);
  const timeoutRef = useRef(null);
  const { stats, loading, error } = useDatasetStats();

  const handleMouseEnter = (tooltipType) => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }
    setActiveTooltip(tooltipType);
  };

  const handleMouseLeave = () => {
    timeoutRef.current = setTimeout(() => {
      setActiveTooltip(null);
    }, 300);
  };

  // Format number with commas
  const formatNumber = (num) => {
    return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
  };

  const tooltipContent = {
    llm: {
      en: (
        <>
          <div>
            <h5 className="font-medium text-gray-600">Technology Overview</h5>
            <p className="text-gray-500 mt-1">Large Language Model (GPT) based analysis that uses advanced AI to understand emotional context in text.</p>
          </div>
          <div>
            <h5 className="font-medium text-gray-600">How it Works</h5>
            <ul className="list-disc ml-4 space-y-1 mt-1 text-gray-500">
              <li>Uses OpenAI's GPT model for deep contextual understanding</li>
              <li>Processes text using transformer architecture</li>
              <li>Analyzes context and semantic relationships</li>
              <li>Generates human-like understanding of emotions</li>
            </ul>
          </div>
          <div>
            <h5 className="font-medium text-gray-600">Advantages</h5>
            <ul className="list-disc ml-4 space-y-1 mt-1 text-gray-500">
              <li>Excellent at understanding complex emotional nuances</li>
              <li>Can handle varied writing styles and contexts</li>
              <li>Adapts to new expressions and situations</li>
              <li>High accuracy in ambiguous cases</li>
            </ul>
          </div>
          <div>
            <h5 className="font-medium text-gray-600">Disadvantages</h5>
            <ul className="list-disc ml-4 space-y-1 mt-1 text-gray-500">
              <li>Requires internet connection and API key</li>
              <li>Higher latency due to API calls</li>
              <li>Can be more expensive for high volume</li>
              <li>Less predictable in edge cases</li>
            </ul>
          </div>
        </>
      ),
      he: (
        <>
          <div>
            <h5 className="font-medium text-gray-600">סקירה טכנולוגית</h5>
            <p className="text-gray-500 mt-1">ניתוח מבוסס מודל שפה גדול (GPT) המשתמש בבינה מלאכותית מתקדמת להבנת הקשר רגשי בטקסט.</p>
          </div>
          <div>
            <h5 className="font-medium text-gray-600">איך זה עובד</h5>
            <ul className="list-disc ml-4 space-y-1 mt-1 text-gray-500">
              <li>שימוש במודל GPT של OpenAI להבנה הקשרית עמוקה</li>
              <li>עיבוד טקסט באמצעות ארכיטקטורת טרנספורמר</li>
              <li>ניתוח הקשר ויחסים סמנטיים</li>
              <li>מייצר הבנה דמוית אנוש של רגשות</li>
            </ul>
          </div>
          <div>
            <h5 className="font-medium text-gray-600">יתרונות</h5>
            <ul className="list-disc ml-4 space-y-1 mt-1 text-gray-500">
              <li>מצוין בהבנת ניואנסים רגשיים מורכבים</li>
              <li>יכול להתמודד עם סגנונות כתיבה והקשרים שונים</li>
              <li>מסתגל לביטויים ומצבים חדשים</li>
              <li>דיוק גבוה במקרים עמומים</li>
            </ul>
          </div>
          <div>
            <h5 className="font-medium text-gray-600">חסרונות</h5>
            <ul className="list-disc ml-4 space-y-1 mt-1 text-gray-500">
              <li>דורש חיבור לאינטרנט ומפתח API</li>
              <li>זמן תגובה גבוה יותר בגלל קריאות API</li>
              <li>יקר יותר עבור נפח גבוה</li>
              <li>פחות צפוי במקרי קצה</li>
            </ul>
          </div>
        </>
      )
    },
    nlp: {
      en: (
        <>
          <div>
            <h5 className="font-medium text-gray-600">Technology Overview</h5>
            <p className="text-gray-500 mt-1">Rule-based Natural Language Processing system using linguistic patterns and emotional lexicons.</p>
          </div>
          <div>
            <h5 className="font-medium text-gray-600">How it Works</h5>
            <ul className="list-disc ml-4 space-y-1 mt-1 text-gray-500">
              <li>Pattern matching with predefined emotional markers</li>
              <li>Lexicon-based emotional word detection</li>
              <li>Contextual rule application</li>
              <li>Syntactic and semantic analysis</li>
            </ul>
          </div>
          <div>
            <h5 className="font-medium text-gray-600">Advantages</h5>
            <ul className="list-disc ml-4 space-y-1 mt-1 text-gray-500">
              <li>Fast and deterministic processing</li>
              <li>No external API dependencies</li>
              <li>Transparent decision-making process</li>
              <li>Consistent performance</li>
            </ul>
          </div>
          <div>
            <h5 className="font-medium text-gray-600">Disadvantages</h5>
            <ul className="list-disc ml-4 space-y-1 mt-1 text-gray-500">
              <li>Limited to predefined patterns</li>
              <li>May miss complex emotional contexts</li>
              <li>Requires manual rule updates</li>
              <li>Less flexible with new expressions</li>
            </ul>
          </div>
        </>
      ),
      he: (
        <>
          <div>
            <h5 className="font-medium text-gray-600">סקירה טכנולוגית</h5>
            <p className="text-gray-500 mt-1">מערכת עיבוד שפה טבעית מבוססת חוקים המשתמשת בדפוסים לשוניים ומילונים רגשיים.</p>
          </div>
          <div>
            <h5 className="font-medium text-gray-600">איך זה עובד</h5>
            <ul className="list-disc ml-4 space-y-1 mt-1 text-gray-500">
              <li>התאמת דפוסים עם סמנים רגשיים מוגדרים מראש</li>
              <li>זיהוי מילים רגשיות מבוסס מילון</li>
              <li>יישום חוקים הקשריים</li>
              <li>ניתוח תחבירי וסמנטי</li>
            </ul>
          </div>
          <div>
            <h5 className="font-medium text-gray-600">יתרונות</h5>
            <ul className="list-disc ml-4 space-y-1 mt-1 text-gray-500">
              <li>עיבוד מהיר ודטרמיניסטי</li>
              <li>ללא תלות ב-API חיצוני</li>
              <li>תהליך קבלת החלטות שקוף</li>
              <li>ביצועים עקביים</li>
            </ul>
          </div>
          <div>
            <h5 className="font-medium text-gray-600">חסרונות</h5>
            <ul className="list-disc ml-4 space-y-1 mt-1 text-gray-500">
              <li>מוגבל לדפוסים מוגדרים מראש</li>
              <li>עלול להחמיץ הקשרים רגשיים מורכבים</li>
              <li>דורש עדכוני חוקים ידניים</li>
              <li>פחות גמיש עם ביטויים חדשים</li>
            </ul>
          </div>
        </>
      )
    },
    custom: {
      en: (
        <>
          <div>
            <h5 className="font-medium text-gray-600">Technology Overview</h5>
            <p className="text-gray-500 mt-1">Fine-tuned BERT model trained on emotional content</p>
          </div>
          <div>
            <h5 className="font-medium text-gray-600">How it Works</h5>
            <ul className="list-disc ml-4 space-y-1 mt-1 text-gray-500">
              <li>Fine-tuned BERT transformer model</li>
              <li>Trained on medical notes and records</li>
              <li>Contextual embeddings for medical terms</li>
              <li>Domain-specific emotion detection</li>
            </ul>
          </div>
          <div>
            <h5 className="font-medium text-gray-600">Advantages</h5>
            <ul className="list-disc ml-4 space-y-1 mt-1 text-gray-500">
              <li>Specialized for medical context</li>
              <li>Fast local inference (~100ms)</li>
              <li>High accuracy (F1: 0.89)</li>
              <li>No API dependencies</li>
            </ul>
          </div>
          <div>
            <h5 className="font-medium text-gray-600">Disadvantages</h5>
            <ul className="list-disc ml-4 space-y-1 mt-1 text-gray-500">
              <li>Limited to training data scope</li>
              <li>Requires periodic retraining</li>
              <li>May overfit to medical domain</li>
              <li>Higher memory requirements</li>
            </ul>
          </div>
        </>
      ),
      he: (
        <>
          <div>
            <h5 className="font-medium text-gray-600">סקירה טכנולוגית</h5>
            <p className="text-gray-500 mt-1">מודל BERT מותאם אישית שאומן במיוחד לניתוח תוכן רגשי בהקשר רפואי.</p>
          </div>
          <div>
            <h5 className="font-medium text-gray-600">איך זה עובד</h5>
            <ul className="list-disc ml-4 space-y-1 mt-1 text-gray-500">
              <li>מודל טרנספורמר BERT מכוונן</li>
              <li>מאומן על רשומות ותיעוד רפואי</li>
              <li>הטמעות הקשריות למונחים רפואיים</li>
              <li>זיהוי רגשות מותאם לתחום</li>
            </ul>
          </div>
          <div>
            <h5 className="font-medium text-gray-600">יתרונות</h5>
            <ul className="list-disc ml-4 space-y-1 mt-1 text-gray-500">
              <li>מותאם במיוחד להקשר רפואי</li>
              <li>הסקה מקומית מהירה (~100ms)</li>
              <li>דיוק גבוה (F1: 0.89)</li>
              <li>ללא תלות ב-API</li>
            </ul>
          </div>
          <div>
            <h5 className="font-medium text-gray-600">חסרונות</h5>
            <ul className="list-disc ml-4 space-y-1 mt-1 text-gray-500">
              <li>מוגבל להיקף נתוני האימון</li>
              <li>דורש אימון מחדש תקופתי</li>
              <li>עלול להתאים יתר על המידה לתחום הרפואי</li>
              <li>דרישות זיכרון גבוהות יותר</li>
            </ul>
          </div>
        </>
      )
    },
    training: {
      en: (
        <>
          <div>
            <h5 className="font-medium text-gray-600">Training Statistics</h5>
            {loading ? (
              <div className="text-center py-2 text-gray-500">Loading statistics...</div>
            ) : error ? (
              <div className="text-center py-2 text-red-500">Error loading statistics</div>
            ) : (
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span className="text-gray-600">Total sentences:</span>
                  <span className="font-medium">{formatNumber(stats.totalSentences)}</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-gray-600">Total labels:</span>
                  <span className="font-medium">{formatNumber(stats.totalLabels)}</span>
                </div>
                <div className="mt-2 pt-2 border-t border-gray-100">
                  <h6 className="text-xs font-medium text-gray-500 mb-1">Label Distribution</h6>
                  <div className="space-y-1">
                    <div className="flex justify-between text-sm">
                      <span className="text-green-600">Progress</span>
                      <span>{formatNumber(stats.distribution.emotional_progress.count)} ({stats.distribution.emotional_progress.percentage}%)</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-yellow-600">Distress</span>
                      <span>{formatNumber(stats.distribution.emotional_distress.count)} ({stats.distribution.emotional_distress.percentage}%)</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-red-600">Danger</span>
                      <span>{formatNumber(stats.distribution.danger.count)} ({stats.distribution.danger.percentage}%)</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-purple-600">Intense</span>
                      <span>{formatNumber(stats.distribution.emotionally_intense.count)} ({stats.distribution.emotionally_intense.percentage}%)</span>
                    </div>
                  </div>
                </div>
                <a 
                  href="/data/evaluation_data.json" 
                  download="evaluation_data.json"
                  className="mt-2 block text-center text-sm text-blue-500 hover:text-blue-700 bg-blue-50 py-1 rounded-md transition-colors"
                  onClick={(e) => {
                    e.stopPropagation();
                  }}
                >
                  <span className="inline-flex items-center justify-center">
                    <ArrowDownTrayIcon className="w-4 h-4 mr-1" />
                    Download Dataset
                  </span>
                </a>
              </div>
            )}
          </div>
        </>
      ),
      he: (
        <>
          <div>
            <h5 className="font-medium text-gray-600">סטטיסטיקות אימון</h5>
            {loading ? (
              <div className="text-center py-2 text-gray-500">טוען סטטיסטיקות...</div>
            ) : error ? (
              <div className="text-center py-2 text-red-500">שגיאה בטעינת סטטיסטיקות</div>
            ) : (
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span className="text-gray-600">סה"כ משפטים:</span>
                  <span className="font-medium">{formatNumber(stats.totalSentences)}</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-gray-600">סה"כ תוויות:</span>
                  <span className="font-medium">{formatNumber(stats.totalLabels)}</span>
                </div>
                <div className="mt-2 pt-2 border-t border-gray-100">
                  <h6 className="text-xs font-medium text-gray-500 mb-1">התפלגות תוויות</h6>
                  <div className="space-y-1">
                    <div className="flex justify-between text-sm">
                      <span className="text-green-600">התקדמות</span>
                      <span>{formatNumber(stats.distribution.emotional_progress.count)} ({stats.distribution.emotional_progress.percentage}%)</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-yellow-600">מצוקה</span>
                      <span>{formatNumber(stats.distribution.emotional_distress.count)} ({stats.distribution.emotional_distress.percentage}%)</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-red-600">סכנה</span>
                      <span>{formatNumber(stats.distribution.danger.count)} ({stats.distribution.danger.percentage}%)</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-purple-600">עוצמה</span>
                      <span>{formatNumber(stats.distribution.emotionally_intense.count)} ({stats.distribution.emotionally_intense.percentage}%)</span>
                    </div>
                  </div>
                </div>
                <a 
                  href="/data/evaluation_data.json" 
                  download="evaluation_data.json"
                  className="mt-2 block text-center text-sm text-blue-500 hover:text-blue-700 bg-blue-50 py-1 rounded-md transition-colors"
                  onClick={(e) => {
                    e.stopPropagation();
                  }}
                >
                  <span className="inline-flex items-center justify-center">
                    <ArrowDownTrayIcon className="w-4 h-4 ml-1" />
                    הורדת מערך נתונים
                  </span>
                </a>
              </div>
            )}
          </div>
        </>
      )
    }
  };

  return (
    <div className="fixed bottom-4 right-4 z-50">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className={`
          inline-flex items-center gap-1.5 
          bg-[#4c8bf5] hover:bg-[#4081f5] 
          text-white text-sm font-medium 
          px-4 py-2 rounded-full shadow-lg
          transition-all duration-300
          hover:scale-105
          ${!isOpen && 'animate-pulse-subtle'}
          hover:shadow-[0_0_15px_rgba(76,139,245,0.5)]
        `}
      >
        <InformationCircleIcon className="w-5 h-5" />
        About
      </button>

      {isOpen && (
        <div className="absolute bottom-12 right-0 w-96 bg-white rounded-xl shadow-lg transform transition-all duration-200">
          <div className="p-4 border-b">
            <div className="flex justify-between items-center">
              <h2 className="text-lg font-semibold text-gray-800">About the Emotion Analyzer</h2>
              <button
                onClick={() => setIsOpen(!isOpen)}
                className="text-gray-400 hover:text-gray-600"
              >
                ✕
              </button>
            </div>
          </div>
          
          <div className="p-4 space-y-4">
            <div>
              <h3 className="font-medium text-gray-700 mb-2">Emotion Labels</h3>
              <ul className="space-y-2">
                <li className="flex items-start gap-2">
                  <span className="bg-red-100 text-red-800 px-2 py-0.5 rounded text-sm font-medium">DANGER</span>
                  <span className="text-gray-600 text-sm">Expressions of self-harm or suicide ideation</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="bg-yellow-100 text-yellow-800 px-2 py-0.5 rounded text-sm font-medium">DISTRESS</span>
                  <span className="text-gray-600 text-sm">Feelings of anxiety, depression, or emotional pain</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="bg-green-100 text-green-800 px-2 py-0.5 rounded text-sm font-medium">PROGRESS</span>
                  <span className="text-gray-600 text-sm">Positive changes, hope, or improvement</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="bg-purple-100 text-purple-800 px-2 py-0.5 rounded text-sm font-medium">INTENSE</span>
                  <span className="text-gray-600 text-sm">Strong emotional expressions or emphasis</span>
                </li>
              </ul>
            </div>

            <div>
              <h3 className="font-medium text-gray-700 mb-2">Analysis Approaches</h3>
              <ul className="space-y-3">
                <li className="flex items-start gap-2">
                  <span className="bg-blue-100 text-blue-800 px-2 py-0.5 rounded text-sm font-medium">LLM</span>
                  <span className="text-gray-600 text-sm group">
                    <div className="flex items-center gap-2">
                      <p>Uses OpenAI's GPT model for nuanced understanding</p>
                      <div className="relative">
                        <button
                          onMouseEnter={() => handleMouseEnter('llm')}
                          onMouseLeave={handleMouseLeave}
                          className="text-blue-500 hover:text-blue-700 transition-colors"
                        >
                          <InformationCircleIcon className="w-4 h-4" />
                        </button>
                        {activeTooltip === 'llm' && (
                          <TechTooltip
                            isHebrew={isHebrew}
                            content={tooltipContent.llm[isHebrew ? 'he' : 'en']}
                            onMouseEnter={() => handleMouseEnter('llm')}
                            onMouseLeave={handleMouseLeave}
                            onLanguageToggle={() => setIsHebrew(!isHebrew)}
                          />
                        )}
                      </div>
                    </div>
                  </span>
                </li>

                <li className="flex items-start gap-2">
                  <span className="bg-blue-100 text-blue-800 px-2 py-0.5 rounded text-sm font-medium">NLP</span>
                  <span className="text-gray-600 text-sm">
                    <div className="flex items-center gap-2">
                      <p>Rule-based approach using predefined patterns</p>
                      <div className="relative">
                        <button
                          onMouseEnter={() => handleMouseEnter('nlp')}
                          onMouseLeave={handleMouseLeave}
                          className="text-blue-500 hover:text-blue-700 transition-colors"
                        >
                          <InformationCircleIcon className="w-4 h-4" />
                        </button>
                        {activeTooltip === 'nlp' && (
                          <TechTooltip
                            isHebrew={isHebrew}
                            content={tooltipContent.nlp[isHebrew ? 'he' : 'en']}
                            onMouseEnter={() => handleMouseEnter('nlp')}
                            onMouseLeave={handleMouseLeave}
                            onLanguageToggle={() => setIsHebrew(!isHebrew)}
                          />
                        )}
                      </div>
                    </div>
                  </span>
                </li>

                <li className="flex items-start gap-2">
                  <span className="bg-blue-100 text-blue-800 px-2 py-0.5 rounded text-sm font-medium">Custom</span>
                  <span className="text-gray-600 text-sm">
                    <div className="flex items-center gap-2">
                      <p>Fine-tuned BERT model trained on emotional content</p>
                      <div className="relative flex items-center gap-2">
                        <button
                          onMouseEnter={() => handleMouseEnter('custom')}
                          onMouseLeave={handleMouseLeave}
                          className="text-blue-500 hover:text-blue-700 transition-colors"
                        >
                          <InformationCircleIcon className="w-4 h-4" />
                        </button>
                        <button
                          onMouseEnter={() => handleMouseEnter('training')}
                          onMouseLeave={handleMouseLeave}
                          className="text-blue-500 hover:text-blue-700 transition-colors"
                        >
                          <ChartBarIcon className="w-4 h-4" />
                        </button>
                        {activeTooltip === 'custom' && (
                          <TechTooltip
                            isHebrew={isHebrew}
                            content={tooltipContent.custom[isHebrew ? 'he' : 'en']}
                            onMouseEnter={() => handleMouseEnter('custom')}
                            onMouseLeave={handleMouseLeave}
                            onLanguageToggle={() => setIsHebrew(!isHebrew)}
                          />
                        )}
                        {activeTooltip === 'training' && (
                          <TechTooltip
                            isHebrew={isHebrew}
                            content={tooltipContent.training[isHebrew ? 'he' : 'en']}
                            onMouseEnter={() => handleMouseEnter('training')}
                            onMouseLeave={handleMouseLeave}
                            onLanguageToggle={() => setIsHebrew(!isHebrew)}
                            type="training"
                          />
                        )}
                      </div>
                    </div>
                  </span>
                </li>
              </ul>
            </div>

            <p className="text-xs text-gray-500 italic">
              Each approach has its strengths and may detect different aspects of emotional content.
            </p>
          </div>
        </div>
      )}
    </div>
  );
};

export default InfoPanel; 