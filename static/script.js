document.addEventListener("DOMContentLoaded", () => {
  const jdUpload = document.getElementById("jdUpload");
  const jdDrop = document.getElementById("jdDrop");
  const jdTextarea = document.getElementById("jdTextarea");
  const includeBox = document.getElementById("includeBox");
  const excludeBox = document.getElementById("excludeBox");
  const generateBoolean = document.getElementById("generateBoolean");
  const booleanQuery = document.getElementById("booleanQuery");
  const resumeUpload = document.getElementById("resumeUpload");
  const resumeDrop = document.getElementById("resumeDrop");
  const matchScore = document.getElementById("matchScore");
  const targetMatch = document.getElementById("targetMatch");
  const targetMatchValue = document.getElementById("targetMatchValue");
  const generatePointers = document.getElementById("generatePointers");
  const cvSuggestions = document.getElementById("cvSuggestions");
  const classificationInfo = document.getElementById("classificationInfo");

  let excludeKeywords = [];
  let uploadedResumeFile = null;
  let storedJDText = "";

  // 📁 JD Upload
  jdUpload.addEventListener("change", handleJDFileUpload);
  jdDrop.addEventListener("click", () => jdUpload.click());
  jdDrop.addEventListener("dragover", (e) => { e.preventDefault(); jdDrop.style.borderColor = "#6c63ff"; });
  jdDrop.addEventListener("dragleave", () => jdDrop.style.borderColor = "#ccc");
  jdDrop.addEventListener("drop", (e) => {
    e.preventDefault();
    jdDrop.style.borderColor = "#ccc";
    jdUpload.files = e.dataTransfer.files;
    handleJDFileUpload();
  });

  async function handleJDFileUpload() {
    const file = jdUpload.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    const res = await fetch("/extract-jd", {
      method: "POST",
      body: formData
    });

    const data = await res.json();
    storedJDText = data.text || "";
    renderKeywordGroups(data.groups, data.exclude);
  }

  // ✍️ JD Text Input
  jdTextarea.addEventListener("blur", async () => {
    const text = jdTextarea.value.trim();
    if (!text) return;

    const res = await fetch("/extract-jd-text", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text })
    });

    const data = await res.json();
    storedJDText = text;
    renderKeywordGroups(data.groups, data.exclude);
  });

  // 🧠 Render Groups
  function renderKeywordGroups(groupData, excludeData) {
    includeBox.innerHTML = "";
    excludeBox.innerHTML = "";
    excludeKeywords = [...excludeData];

    groupData.forEach(group => {
      const groupBox = createGroupBox(group);
      includeBox.appendChild(groupBox);
    });

    excludeData.forEach(kw => {
      const span = document.createElement("span");
      span.className = "keyword exclude";
      span.innerText = kw;
      excludeBox.appendChild(span);
    });

    updateGroupsFromDOM();
    refreshGroupTitles();
  }

  // ➕ Create Group Box
  function createGroupBox(keywords = []) {
    const groupBox = document.createElement("div");
    groupBox.className = "group-box";

    const groupTitle = document.createElement("div");
    groupTitle.className = "group-title";
    groupBox.appendChild(groupTitle);

    const groupList = document.createElement("div");
    groupList.className = "sortable-group";
    groupBox.appendChild(groupList);

    keywords.forEach(kw => {
      const span = createKeywordTag(kw);
      groupList.appendChild(span);
    });

    const addBtn = document.createElement("div");
    addBtn.className = "keyword-input-btn";
    addBtn.innerText = "➕ Add Keyword";
    addBtn.onclick = () => showInlineInput(addBtn, groupList);
    groupBox.appendChild(addBtn);

    Sortable.create(groupList, {
      group: "shared",
      animation: 150,
      onEnd: () => {
        updateGroupsFromDOM();
      }
    });

    return groupBox;
  }

  function createKeywordTag(value) {
    const span = document.createElement("span");
    span.className = "keyword";
    span.innerHTML = `${value} <span class="remove-btn">×</span>`;
    attachKeywordEvents(span);
    return span;
  }

  function attachKeywordEvents(span) {
    span.querySelector(".remove-btn").addEventListener("click", (e) => {
      e.stopPropagation();
      span.remove();
      updateGroupsFromDOM();
    });

    span.childNodes[0].addEventListener("dblclick", () => {
      const current = span.childNodes[0].textContent.trim();
      const input = document.createElement("input");
      input.type = "text";
      input.value = current;
      input.className = "keyword";

      span.innerHTML = "";
      span.appendChild(input);
      input.focus();

      const save = () => {
        const newVal = input.value.trim();
        if (newVal) {
          span.innerHTML = `${newVal} <span class="remove-btn">×</span>`;
          attachKeywordEvents(span);
          updateGroupsFromDOM();
        } else {
          span.remove();
          updateGroupsFromDOM();
        }
      };

      input.addEventListener("keydown", (e) => {
        if (e.key === "Enter") save();
        else if (e.key === "Escape") {
          span.innerHTML = `${current} <span class="remove-btn">×</span>`;
          attachKeywordEvents(span);
        }
      });

      input.addEventListener("blur", save);
    });
  }

  function showInlineInput(addBtn, container) {
    const input = document.createElement("input");
    input.type = "text";
    input.placeholder = "Enter keyword";
    input.className = "keyword";
    addBtn.before(input);
    input.focus();

    const save = () => {
      const val = input.value.trim();
      if (val) {
        const span = createKeywordTag(val);
        input.remove();
        addBtn.before(span);
        updateGroupsFromDOM();
      } else {
        input.remove();
      }
    };

    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter") save();
      else if (e.key === "Escape") input.remove();
    });

    input.addEventListener("blur", save);
  }

  function updateGroupsFromDOM() {
    const groupDivs = document.querySelectorAll(".sortable-group");
    window.groups = [];

    groupDivs.forEach(groupEl => {
      const keywords = Array.from(groupEl.querySelectorAll(".keyword")).map(k => {
        const textNode = k.childNodes[0];
        return textNode ? textNode.textContent.trim() : '';
      }).filter(Boolean);

      if (keywords.length) window.groups.push(keywords);
    });
  }

  function refreshGroupTitles() {
    const groupBoxes = document.querySelectorAll("#includeBox .group-box");

    groupBoxes.forEach((box, index) => {
      const title = box.querySelector(".group-title");
      title.innerHTML = `Group ${index + 1} <span class="remove-group">×</span>`;
      const btn = title.querySelector(".remove-group");

      btn.addEventListener("click", () => {
        box.remove();
        updateGroupsFromDOM();
        refreshGroupTitles();
      });
    });
  }

  window.addNewGroup = function () {
    const groupBox = createGroupBox([]);
    includeBox.appendChild(groupBox);
    updateGroupsFromDOM();
    refreshGroupTitles();
  };

  generateBoolean.addEventListener("click", async () => {
    updateGroupsFromDOM();

    const res = await fetch("/generate-boolean", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        groups: window.groups || [],
        exclude: excludeKeywords
      })
    });

    const data = await res.json();
    booleanQuery.value = data.boolean_query;
  });

  resumeUpload.addEventListener("change", () => {
    uploadedResumeFile = resumeUpload.files[0];
  });

  resumeDrop.addEventListener("click", () => resumeUpload.click());
  resumeDrop.addEventListener("dragover", (e) => { e.preventDefault(); resumeDrop.style.borderColor = "#6c63ff"; });
  resumeDrop.addEventListener("dragleave", () => resumeDrop.style.borderColor = "#ccc");
  resumeDrop.addEventListener("drop", (e) => {
    e.preventDefault();
    resumeDrop.style.borderColor = "#ccc";
    resumeUpload.files = e.dataTransfer.files;
    uploadedResumeFile = resumeUpload.files[0];
  });

  targetMatch.addEventListener("input", () => {
    targetMatchValue.innerText = `${targetMatch.value}%`;
  });

  generatePointers.addEventListener("click", async () => {
    if (!uploadedResumeFile) return;

    const formData = new FormData();
    formData.append("resume", uploadedResumeFile);
    formData.append("target_match", targetMatch.value);
    formData.append("jd_text", storedJDText);

    const res = await fetch("/generate-pointers", {
      method: "POST",
      body: formData
    });

    const data = await res.json();
    matchScore.value = `${data.match_score}%`;

    classificationInfo.innerHTML = `
      <strong>ROLE:</strong> ${data.role_title} 
      <strong>TYPE:</strong> ${data.role_type} 
      <strong>LEVEL:</strong> ${data.level}
    `;

    cvSuggestions.innerText = data.updated_pointers.join("\n\n");
  });
});
