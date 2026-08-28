

document.addEventListener('DOMContentLoaded', () => {

    const systemStatusEl = document.getElementById('systemStatus');
    const statusTextEl = document.getElementById('statusText');
    const samplePresetsEl = document.getElementById('samplePresets');

    const promptTextEl = document.getElementById('promptText');
    const structuredDataEl = document.getElementById('structuredData');
    const referenceTextEl = document.getElementById('referenceText');

    const styleCards = document.querySelectorAll('.style-card');


    const maxTokensEl = document.getElementById('maxTokens');
    const maxTokensValEl = document.getElementById('maxTokensVal');
    const numBeamsEl = document.getElementById('numBeams');
    const numBeamsValEl = document.getElementById('numBeamsVal');
    const tempEl = document.getElementById('temperature');
    const tempValEl = document.getElementById('tempVal');
    const repEl = document.getElementById('repetitionPenalty');
    const repValEl = document.getElementById('repVal');


    const btnGenerate = document.getElementById('btnGenerate');
    const btnClear = document.getElementById('btnClear');
    const btnCopy = document.getElementById('btnCopy');
    const btnDownloadTxt = document.getElementById('btnDownloadTxt');
    const btnDownloadJson = document.getElementById('btnDownloadJson');
    const btnClearHistory = document.getElementById('btnClearHistory');


    const loadingIndicator = document.getElementById('loadingIndicator');
    const outputSection = document.getElementById('outputSection');
    const generatedTextContent = document.getElementById('generatedTextContent');
    const genTimeBadge = document.getElementById('genTimeBadge');
    const modelBadge = document.getElementById('modelBadge');


    const statChars = document.getElementById('statChars');
    const statWords = document.getElementById('statWords');
    const statSentences = document.getElementById('statSentences');
    const statAvgWord = document.getElementById('statAvgWord');


    const evaluationContainer = document.getElementById('evaluationContainer');
    const noReferenceNotice = document.getElementById('noReferenceNotice');
    const evaluationPanel = document.getElementById('evaluationPanel');

    const valRouge1 = document.getElementById('valRouge1');
    const barRouge1 = document.getElementById('barRouge1');
    const valRouge2 = document.getElementById('valRouge2');
    const barRouge2 = document.getElementById('barRouge2');
    const valRougeL = document.getElementById('valRougeL');
    const barRougeL = document.getElementById('barRougeL');

    const valBleu1 = document.getElementById('valBleu1');
    const barBleu1 = document.getElementById('barBleu1');
    const valBleu2 = document.getElementById('valBleu2');
    const barBleu2 = document.getElementById('barBleu2');
    const valBleu3 = document.getElementById('valBleu3');
    const barBleu3 = document.getElementById('barBleu3');
    const valBleu4 = document.getElementById('valBleu4');
    const barBleu4 = document.getElementById('barBleu4');

    const historyContainer = document.getElementById('historyContainer');
    const toastContainer = document.getElementById('toastContainer');


    let selectedStyle = 'general';
    let currentResultData = null;


    const PRESETS = {
        cs_dept: {
            prompt: "",
            structured: "Department: Computer Science\nStudents: 120\nAverage Attendance: 91%\nPass Percentage: 94%\nTop Subject: Data Structures",
            reference: "The Computer Science Department currently serves 120 students with an impressive average attendance rate of 91% and a 94% pass percentage. Data Structures remains the top performing subject."
        },
        financial: {
            prompt: "",
            structured: "Revenue: ₹12.5M\nGrowth: 18%\nCustomers: 24,500\nChurn: 4.2%",
            reference: "The company achieved revenue of ₹12.5M with an 18% growth rate. Total active customers reached 24,500 with a low churn rate of 4.2%."
        },
        prompt_cs: {
            prompt: "Write a short formal description of a university computer science department.",
            structured: "",
            reference: "The Department of Computer Science offers comprehensive undergraduate and postgraduate programs focused on computational theory, software engineering, artificial intelligence, and data analytics."
        },
        prompt_product: {
            prompt: "Write a descriptive summary announcing a new AI cloud platform for software developers.",
            structured: "",
            reference: "We are excited to launch our next-generation AI Cloud Platform engineered for modern developers. It delivers automated model training, seamless API integration, and enterprise-grade performance."
        }
    };


    fetchHealthStatus();
    fetchHistory();

    function fetchHealthStatus() {
        fetch('/health')
            .then(res => res.json())
            .then(data => {
                if (data.status === 'healthy') {
                    systemStatusEl.classList.add('online');
                    statusTextEl.textContent = `Model: ${data.model_name} (${data.device.toUpperCase()})`;
                    modelBadge.textContent = data.model_name;
                }
            })
            .catch(() => {
                statusTextEl.textContent = 'Backend Service Disconnected';
            });
    }


    maxTokensEl.addEventListener('input', (e) => maxTokensValEl.textContent = e.target.value);
    numBeamsEl.addEventListener('input', (e) => numBeamsValEl.textContent = e.target.value);
    tempEl.addEventListener('input', (e) => tempValEl.textContent = e.target.value);
    repEl.addEventListener('input', (e) => repValEl.textContent = e.target.value);


    styleCards.forEach(card => {
        card.addEventListener('click', () => selectStyleCard(card));
        card.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                selectStyleCard(card);
            }
        });
    });

    function selectStyleCard(card) {
        styleCards.forEach(c => {
            c.classList.remove('active');
            c.setAttribute('aria-checked', 'false');
        });
        card.classList.add('active');
        card.setAttribute('aria-checked', 'true');
        selectedStyle = card.dataset.style;
    }


    samplePresetsEl.addEventListener('change', (e) => {
        const presetKey = e.target.value;
        if (PRESETS[presetKey]) {
            promptTextEl.value = PRESETS[presetKey].prompt;
            structuredDataEl.value = PRESETS[presetKey].structured;
            referenceTextEl.value = PRESETS[presetKey].reference;
            showToast('Sample preset loaded.', 'success');
        }
    });


    btnClear.addEventListener('click', () => {
        promptTextEl.value = '';
        structuredDataEl.value = '';
        referenceTextEl.value = '';
        samplePresetsEl.value = '';
        outputSection.classList.add('hidden');
        showToast('Input fields cleared.', 'success');
    });


    btnGenerate.addEventListener('click', () => {
        const promptText = promptTextEl.value.trim();
        const structuredData = structuredDataEl.value.trim();
        const referenceText = referenceTextEl.value.trim();

        if (!promptText && !structuredData) {
            showToast('Please provide either a prompt instruction or structured data.', 'error');
            return;
        }

        const payload = {
            prompt_text: promptText,
            structured_data: structuredData,
            reference_text: referenceText,
            style: selectedStyle,
            max_new_tokens: parseInt(maxTokensEl.value),
            num_beams: parseInt(numBeamsEl.value),
            temperature: parseFloat(tempEl.value),
            repetition_penalty: parseFloat(repEl.value)
        };


        const btnTextEl = btnGenerate.querySelector('.btn-text');
        btnGenerate.disabled = true;
        if (btnTextEl) btnTextEl.textContent = 'Generating...';

        outputSection.classList.add('hidden');
        loadingIndicator.classList.remove('hidden');

        fetch('/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        })
        .then(res => {
            if (!res.ok) {
                return res.json().then(errData => {
                    throw new Error(errData.error || `Server error (${res.status})`);
                });
            }
            return res.json();
        })
        .then(data => {
            loadingIndicator.classList.add('hidden');
            btnGenerate.disabled = false;
            if (btnTextEl) btnTextEl.textContent = 'Generate Natural Text';

            if (!data.success) {
                showToast(data.error || 'Text generation failed.', 'error');
                return;
            }

            currentResultData = data;
            renderOutput(data);
            fetchHistory();
            showToast('Text generated successfully!', 'success');
        })
        .catch(err => {
            loadingIndicator.classList.add('hidden');
            btnGenerate.disabled = false;
            if (btnTextEl) btnTextEl.textContent = 'Generate Natural Text';
            showToast(err.message || 'Error connecting to backend server.', 'error');
        });
    });


    function renderOutput(data) {
        const res = data.result;
        const stats = data.statistics;
        const evalData = data.evaluation;

        generatedTextContent.textContent = res.generated_text;
        genTimeBadge.textContent = `${res.execution_time_sec}s`;
        modelBadge.textContent = res.model_name;


        const factCoverageEl = document.getElementById('factCoverageBadge');
        if (factCoverageEl) {
            const fc = res.fact_coverage || data.fact_coverage;
            if (fc && fc.coverage) {
                const isComplete = fc.complete && fc.semantic_coverage !== false;
                const statusLabel = isComplete ? 'Semantic Fact Coverage' : 'Fact Coverage';
                factCoverageEl.textContent = `${statusLabel}: ${fc.coverage} (${fc.coverage_percentage}%)`;
                factCoverageEl.classList.remove('hidden');
                factCoverageEl.style.backgroundColor = isComplete ? 'rgba(46, 204, 113, 0.2)' : 'rgba(241, 196, 15, 0.2)';
                factCoverageEl.style.color = isComplete ? '#2ecc71' : '#f1c40f';
                factCoverageEl.style.borderColor = isComplete ? '#2ecc71' : '#f1c40f';
            } else {
                factCoverageEl.classList.add('hidden');
            }
        }


        statChars.textContent = stats.char_count;
        statWords.textContent = stats.word_count;
        statSentences.textContent = stats.sentence_count;
        statAvgWord.textContent = stats.avg_word_length;


        if (evalData && evalData.metrics_available && evalData.metrics) {
            noReferenceNotice.classList.add('hidden');
            evaluationPanel.classList.remove('hidden');
            const metrics = evalData.metrics;

            if (metrics.rouge) {
                const r1 = metrics.rouge.rouge1.fmeasure;
                const r2 = metrics.rouge.rouge2.fmeasure;
                const rL = metrics.rouge.rougeL.fmeasure;

                valRouge1.textContent = r1.toFixed(4);
                barRouge1.style.width = `${Math.min(100, r1 * 100)}%`;

                valRouge2.textContent = r2.toFixed(4);
                barRouge2.style.width = `${Math.min(100, r2 * 100)}%`;

                valRougeL.textContent = rL.toFixed(4);
                barRougeL.style.width = `${Math.min(100, rL * 100)}%`;
            }

            if (metrics.bleu) {
                const b1 = metrics.bleu.bleu_1;
                const b2 = metrics.bleu.bleu_2;
                const b3 = metrics.bleu.bleu_3 !== undefined ? metrics.bleu.bleu_3 : metrics.bleu.bleu_2;
                const b4 = metrics.bleu.bleu_4;

                valBleu1.textContent = b1.toFixed(4);
                barBleu1.style.width = `${Math.min(100, b1 * 100)}%`;

                valBleu2.textContent = b2.toFixed(4);
                barBleu2.style.width = `${Math.min(100, b2 * 100)}%`;

                if (valBleu3 && barBleu3) {
                    valBleu3.textContent = b3.toFixed(4);
                    barBleu3.style.width = `${Math.min(100, b3 * 100)}%`;
                }

                valBleu4.textContent = b4.toFixed(4);
                barBleu4.style.width = `${Math.min(100, b4 * 100)}%`;
            }
        } else {
            evaluationPanel.classList.add('hidden');
            noReferenceNotice.classList.remove('hidden');
        }

        outputSection.classList.remove('hidden');
        outputSection.scrollIntoView({ behavior: 'smooth' });
    }


    btnCopy.addEventListener('click', () => {
        const text = generatedTextContent.textContent;
        if (!text) return;
        navigator.clipboard.writeText(text)
            .then(() => showToast('Copied to clipboard!', 'success'))
            .catch(() => showToast('Failed to copy text.', 'error'));
    });


    btnDownloadTxt.addEventListener('click', () => {
        const text = generatedTextContent.textContent;
        if (!text) return;
        downloadFile(text, 'generated_text.txt', 'text/plain');
    });

    btnDownloadJson.addEventListener('click', () => {
        if (!currentResultData) return;

        const exportData = {
            generated_text: currentResultData.result.generated_text,
            style: currentResultData.result.style,
            prompt_text: currentResultData.result.prompt_text,
            structured_data: currentResultData.result.structured_data,
            statistics: currentResultData.statistics,
            evaluation: currentResultData.evaluation,
            parameters: currentResultData.result.parameters,
            model_name: currentResultData.result.model_name,
            execution_time_sec: currentResultData.result.execution_time_sec,
            timestamp: new Date().toISOString()
        };

        const jsonStr = JSON.stringify(exportData, null, 2);
        downloadFile(jsonStr, 'generation_report.json', 'application/json');
    });

    function downloadFile(content, filename, type) {
        const blob = new Blob([content], { type: type });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        showToast(`Downloaded ${filename}`, 'success');
    }


    function fetchHistory() {
        fetch('/history')
            .then(res => res.json())
            .then(data => {
                if (data.success) {
                    renderHistory(data.history);
                }
            })
            .catch(err => {
                console.error("Failed to fetch history:", err);
            });
    }

    function renderHistory(history) {
        if (!history || history.length === 0) {
            historyContainer.innerHTML = `
                <div class="empty-msg">
                    <strong>No generations yet.</strong>
                    <p>Generate your first piece of text to see it here.</p>
                </div>`;
            btnClearHistory.classList.add('hidden');
            return;
        }

        btnClearHistory.classList.remove('hidden');

        const totalCount = history.length;
        historyContainer.innerHTML = history.map((item, idx) => {
            const genNum = totalCount - idx;
            const wordCount = item.statistics ? item.statistics.word_count : 0;
            const fullText = item.generated_text || '';
            const isLong = fullText.length > 140;
            const textPreview = isLong ? fullText.substring(0, 140) + '...' : fullText;

            return `
                <div class="history-item" data-id="${item.id}">
                    <div class="history-header">
                        <div class="history-meta">
                            <span class="gen-badge">Generation #${genNum}</span>
                            <span class="style-badge">${escapeHtml(item.style)}</span>
                            <span class="history-time">${escapeHtml(item.timestamp)}</span>
                        </div>
                        <div class="history-stats">
                            <span>${wordCount} words</span>
                            <span>&bull;</span>
                            <span>${item.execution_time_sec}s</span>
                        </div>
                    </div>
                    ${(item.prompt_text || item.structured_data) ? `
                        <div class="history-prompt">
                            <strong>Input:</strong> "${escapeHtml(item.prompt_text || item.structured_data)}"
                        </div>` : ''}
                    <div class="history-text-box">
                        <p class="history-text">${escapeHtml(isLong ? textPreview : fullText)}</p>
                        ${isLong ? `
                            <button class="btn-text-toggle" data-full="${escapeHtml(fullText)}" data-preview="${escapeHtml(textPreview)}" onclick="toggleHistoryText(this)">View Full Text</button>
                        ` : ''}
                    </div>
                </div>
            `;
        }).join('');
    }


    window.toggleHistoryText = function(btn) {
        const textEl = btn.previousElementSibling;
        const full = btn.dataset.full;
        const preview = btn.dataset.preview;

        if (btn.textContent === 'View Full Text') {
            textEl.textContent = full;
            btn.textContent = 'Show Less';
        } else {
            textEl.textContent = preview;
            btn.textContent = 'View Full Text';
        }
    };

    btnClearHistory.addEventListener('click', () => {
        fetch('/clear', { method: 'POST' })
            .then(res => res.json())
            .then(data => {
                if (data.success) {
                    renderHistory([]);
                    showToast('Session history cleared.', 'success');
                }
            });
    });


    function showToast(message, type = 'info') {
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.textContent = message;
        toastContainer.appendChild(toast);
        setTimeout(() => toast.remove(), 3500);
    }

    function escapeHtml(str) {
        if (!str) return '';
        return str.replace(/[&<>"']/g, match => ({
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#039;'
        }[match]));
    }
});

