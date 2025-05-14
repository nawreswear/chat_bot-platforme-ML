// This file ensures proper integration between chatbot.js and the HTML file

// Make sure the document is loaded before checking for functions
document.addEventListener('DOMContentLoaded', () => {
    // Check if simulateMLProcessing exists
    if (typeof simulateMLProcessing !== 'function') {
        // Define a fallback function if it doesn't exist
        window.simulateMLProcessing = async function(message) {
            console.log('Using fallback simulateMLProcessing for:', message);
            
            try {
                document.getElementById('typing').style.display = 'block';
                document.getElementById('loading').style.display = 'block';
                
                // Simple fallback response
                setTimeout(() => {
                    const response = "I'm sorry, but I'm currently operating in fallback mode. The main processing system is unavailable.";
                    const intent = "Fallback";
                    const entities = [];
                    const pageLink = null;
                    const mlInsights = {
                        intent: "Fallback",
                        confidence: 100,
                        entities: [],
                        sentiment: {
                            overall: "neutral",
                            scores: { positive: 0.2, neutral: 0.7, negative: 0.1 }
                        }
                    };
                    
                    window.appendMessage('bot', response, intent, entities, pageLink, mlInsights);
                    
                    document.getElementById('typing').style.display = 'none';
                    document.getElementById('loading').style.display = 'none';
                }, 1000);
                
            } catch (error) {
                console.error('Error in fallback processing:', error);
                document.getElementById('typing').style.display = 'none';
                document.getElementById('loading').style.display = 'none';
            }
        };
        
        console.warn('simulateMLProcessing was not found, using fallback implementation');
    } else {
        console.log('simulateMLProcessing function found and ready');
    }
});