// Declare currentLang, analyticsData, and appendMessage
let currentLang = 'en'; // Default language
let analyticsData = JSON.parse(localStorage.getItem('analyticsData')) || {
    intentDistribution: {},
    sentimentAnalysis: {
        positive: 0,
        neutral: 0,
        negative: 0
    },
    mlAccuracy: 75
};

// This function will be called when a user sends a message
window.simulateMLProcessing = async function(message) {
    console.log('Processing message:', message);

    try {
        // Show typing indicator
        document.getElementById('typing').style.display = 'block';
        document.getElementById('loading').style.display = 'block';
        
        // Make an actual API call to the backend
        const response = await fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message: message,
                language: currentLang
            }),
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        
        // Extract data from the response
        const botResponse = result.response;
        const intent = result.intent || 'General_Query';
        const pageLink = result.page_link;
        const confidence = result.confidence || 0.75;
        
        // Create entities array (this would ideally come from backend in a full implementation)
        let entities = [];
        if (message.toLowerCase().includes('iset') || message.toLowerCase().includes('sfax')) {
            entities.push({ type: 'institution', value: 'ISET Sfax' });
        }
        
        // Determine sentiment (simplified for demo)
        let sentiment = 'neutral';
        let sentimentScores = { positive: 0.2, neutral: 0.7, negative: 0.1 };
        
        if (message.toLowerCase().includes('love') || message.toLowerCase().includes('great')) {
            sentiment = 'positive';
            sentimentScores = { positive: 0.8, neutral: 0.15, negative: 0.05 };
            analyticsData.sentimentAnalysis.positive++;
        } else if (message.toLowerCase().includes('hate') || message.toLowerCase().includes('bad')) {
            sentiment = 'negative';
            sentimentScores = { positive: 0.05, neutral: 0.15, negative: 0.8 };
            analyticsData.sentimentAnalysis.negative++;
        } else {
            analyticsData.sentimentAnalysis.neutral++;
        }
        
        // Update intent distribution in analytics
        analyticsData.intentDistribution[intent] = (analyticsData.intentDistribution[intent] || 0) + 1;
        
        // Update ML accuracy
        analyticsData.mlAccuracy = Math.round(confidence * 100);
        localStorage.setItem('analyticsData', JSON.stringify(analyticsData));
        
        // Create ML insights
        const mlInsights = {
            intent,
            confidence: Math.round(confidence * 100),
            entities,
            sentiment: {
                overall: sentiment,
                scores: sentimentScores
            }
        };
        
        // Append bot message
        window.appendMessage('bot', botResponse, intent, entities, pageLink, mlInsights);
        
    } catch (error) {
        console.error('Error processing message:', error);
        
        // Fallback to local processing if backend is unavailable
        window.fallbackProcessing(message);
    } finally {
        document.getElementById('typing').style.display = 'none';
        document.getElementById('loading').style.display = 'none';
    }
};

// Fallback function when backend is unavailable
window.fallbackProcessing = function(message) {
    console.log('Using fallback processing for:', message);

    // Simple intent classification based on keywords
    let intent = 'General_Query';
    let confidence = 0.75;
    let response = '';
    let pageLink = null;

    const normalizedMessage = message.toLowerCase();

    if (normalizedMessage.includes('course') || normalizedMessage.includes('register') || normalizedMessage.includes('registration')) {
        intent = 'Course_Registration';
        confidence = 0.85;
        response = 'To register for courses at ISET Sfax, you need to log in to the student portal with your student ID and password. The registration period for the current semester is from August 25 to September 10.';
        pageLink = 'https://isetsf.rnu.tn/fr/formations-top';
        analyticsData.intentDistribution['Course_Registration'] = (analyticsData.intentDistribution['Course_Registration'] || 0) + 1;
    } else if (normalizedMessage.includes('calendar') || normalizedMessage.includes('semester') || normalizedMessage.includes('date')) {
        intent = 'Academic_Calendar';
        confidence = 0.9;
        response = 'The current academic year at ISET Sfax runs from September 15 to June 15. The fall semester is from September 15 to January 15, with exams from January 5-15.';
        pageLink = 'https://isetsf.rnu.tn/fr/actualites/manifestations';
        analyticsData.intentDistribution['Academic_Calendar'] = (analyticsData.intentDistribution['Academic_Calendar'] || 0) + 1;
    } else if (normalizedMessage.includes('program') || normalizedMessage.includes('major') || normalizedMessage.includes('degree')) {
        intent = 'Programs';
        confidence = 0.88;
        response = 'ISET Sfax offers various programs including Computer Science Engineering, Electrical Engineering, Mechanical Engineering, Business Administration, and Multimedia Technology.';
        pageLink = 'https://isetsf.rnu.tn/fr/formations-top';
        analyticsData.intentDistribution['Programs'] = (analyticsData.intentDistribution['Programs'] || 0) + 1;
    } else if (normalizedMessage.includes('admission') || normalizedMessage.includes('apply') || normalizedMessage.includes('application')) {
        intent = 'Admission_Requirements';
        confidence = 0.87;
        response = 'Admission to ISET Sfax requires a high school diploma with a minimum GPA of 12/20, with preference given to science and technical backgrounds.';
        pageLink = 'https://isetsf.rnu.tn/fr/formations-top';
        analyticsData.intentDistribution['Admission_Requirements'] = (analyticsData.intentDistribution['Admission_Requirements'] || 0) + 1;
    } else if (normalizedMessage.includes('library') || normalizedMessage.includes('lab') || normalizedMessage.includes('campus')) {
        intent = 'Campus_Facilities';
        confidence = 0.82;
        response = 'ISET Sfax campus includes a central library with over 25,000 books and digital resources, computer labs equipped with the latest software, and specialized engineering laboratories.';
        pageLink = 'https://isetsf.rnu.tn/fr/contactez-nous';
        analyticsData.intentDistribution['Campus_Facilities'] = (analyticsData.intentDistribution['Campus_Facilities'] || 0) + 1;
    } else {
        analyticsData.intentDistribution['Other'] = (analyticsData.intentDistribution['Other'] || 0) + 1;
        response = 'I understand you\'re asking about ISET Sfax. Could you please provide more specific details about what you\'d like to know?';
    }

    // Create entities array
    let entities = [];
    if (normalizedMessage.includes('iset') || normalizedMessage.includes('sfax')) {
        entities.push({ type: 'institution', value: 'ISET Sfax' });
    }

    // Determine sentiment
    let sentiment = 'neutral';
    let sentimentScores = { positive: 0.2, neutral: 0.7, negative: 0.1 };

    if (normalizedMessage.includes('love') || normalizedMessage.includes('great')) {
        sentiment = 'positive';
        sentimentScores = { positive: 0.8, neutral: 0.15, negative: 0.05 };
        analyticsData.sentimentAnalysis.positive++;
    } else if (normalizedMessage.includes('hate') || normalizedMessage.includes('bad')) {
        sentiment = 'negative';
        sentimentScores = { positive: 0.05, neutral: 0.15, negative: 0.8 };
        analyticsData.sentimentAnalysis.negative++;
    } else {
        analyticsData.sentimentAnalysis.neutral++;
    }

    // Update ML accuracy in analytics
    analyticsData.mlAccuracy = Math.round(confidence * 100);
    localStorage.setItem('analyticsData', JSON.stringify(analyticsData));

    // Create ML insights object
    const mlInsights = {
        intent: intent,
        confidence: Math.round(confidence * 100),
        entities: entities,
        sentiment: {
            overall: sentiment,
            scores: sentimentScores
        }
    };

    // Append bot message with ML insights
    window.appendMessage('bot', response, intent, entities, pageLink, mlInsights);
};

console.log('Chatbot.js loaded successfully');
