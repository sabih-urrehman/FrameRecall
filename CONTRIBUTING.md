# Collaborating with FrameRecall

We appreciate your desire to enhance FrameRecall. Every contribution, big or small, is welcome.

## 🚀 Initial Setup

1. **Duplicate the project** via GitHub  
2. **Download your copy** using:
   ```bash
   git clone https://github.com/YOUR_USERNAME/framerecall.git
   cd framerecall
   ```
3. **Activate a Python environment**:
   ```bash
   python -m venv .framerecall
   source .framerecall/bin/activate  # On Windows: .framerecall\Scripts\activate
   ```
4. **Configure all dev packages**:
   ```bash
   pip install -e ".[dev]"
   pip install PyPDF2  # For handling PDFs
   ```

## 🔧 Contribution Guidelines

1. **Spin up a dedicated branch** for modifications:
   ```bash
   git checkout -b enhancement/your-topic
   ```

2. **Apply your updates**, checking that:
   - Syntax matches project conventions  
   - New capabilities include unit checks  
   - Validation passes: `pytest tests/`  
   - Formatting is consistent  

3. **Execute validation suite**:
   ```bash
   pytest tests/
   pytest --cov=framerecall tests/  # Coverage insights
   ```

4. **Record your work**:
   ```bash
   git add .
   git commit -m "Update: brief summary of edits"
   ```

5. **Send changes to your fork**:
   ```bash
   git push origin enhancement/your-topic
   ```

6. **Submit a Merge Request** through GitHub

## 📝 Formatting Conventions

- Adhere to PEP 8 rules  
- Select descriptive identifiers  
- Use docstrings on each method and class  
- Maintain focused, short functions  
- Avoid inactive snippets

## 🧪 Verification

- Incorporate tests for added tools  
- Confirm everything passes prior to merge requests  
- Increase coverage where feasible  
- Include boundary scenarios

## 📚 Writing Docs

- Refresh README.md with any additional utilities  
- Follow Google docstring format  
- Supply examples to demonstrate features  
- Revise CLAUDE.md if structural shifts occur

## 🐛 Bug Reports

When noting problems, provide:
- Python interpreter build  
- OS details  
- Full traceback  
- Instructions to replicate  
- Desired result versus actual output

## 💡 Suggestions

We’re always open to enhancements. If proposing:
- Clarify the need  
- Outline anticipated outcome  
- Ensure legacy support remains intact  
- Suggest a strategy

## 🤝 Community Expectations

- Remain kind and supportive  
- Assist those unfamiliar with the codebase  
- Focus feedback on improvements  
- Embrace a wide range of approaches

## 📞 Need Help?

- File issues for support or additions  
- Participate in open topics  
- Ping @olow304 for quick responses

## ✅ Before Requesting a Merge

- [ ] All tests succeed locally  
- [ ] Follows the style guide  
- [ ] Commits are meaningful  
- [ ] Instructions are current  
- [ ] Explanation of modifications is present  
- [ ] Issue reference is included (if needed)

## 🎉 Appreciation

Contributors receive:
- A spot on our contributor registry  
- Acknowledgement in change logs  
- Inclusion in the outstanding FrameRecall team!

---

**Gratitude for strengthening FrameRecall!** 🙌
