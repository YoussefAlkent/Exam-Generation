import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, LETTER, LEGAL
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.platypus.flowables import PageBreak
from src.generation.assignment_types import Assignment, Deliverable, TechnicalConstraint, PerformanceRequirement
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class PDFGenerator:
    def __init__(self, output_dir: str = "output"):
        """
        Initialize the PDF generator.
        
        Args:
            output_dir: Directory where PDFs will be saved
        """
        print(f"[PDFGenerator] Initializing with output directory: {output_dir}")
        self.output_dir = output_dir
        try:
            os.makedirs(output_dir, exist_ok=True)
            print(f"[PDFGenerator] Created/verified output directory: {output_dir}")
        except Exception as e:
            print(f"[PDFGenerator] ERROR: Failed to create output directory: {str(e)}")
            raise
        
        # Initialize styles
        print("[PDFGenerator] Setting up document styles")
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
        self._register_fonts()
        print("[PDFGenerator] Initialization complete")
        
    def _setup_custom_styles(self):
        """Set up custom paragraph styles."""
        # Update existing styles instead of adding new ones
        self.styles['Heading1'].fontSize = 24
        self.styles['Heading1'].spaceAfter = 30
        self.styles['Heading1'].textColor = colors.HexColor('#2C3E50')
        
        self.styles['Heading2'].fontSize = 18
        self.styles['Heading2'].spaceAfter = 20
        self.styles['Heading2'].textColor = colors.HexColor('#34495E')
        
        self.styles['Heading3'].fontSize = 14
        self.styles['Heading3'].spaceAfter = 15
        self.styles['Heading3'].textColor = colors.HexColor('#7F8C8D')
        
        self.styles['BodyText'].fontSize = 12
        self.styles['BodyText'].spaceAfter = 12
        self.styles['BodyText'].leading = 14
        
        # Add a new style for code blocks
        self.styles.add(ParagraphStyle(
            name='CodeBlock',
            parent=self.styles['Code'],
            fontName='Courier',
            fontSize=10,
            leading=12,
            backColor=colors.HexColor('#F8F9FA'),
            borderColor=colors.HexColor('#DEE2E6'),
            borderWidth=1,
            borderPadding=5,
            borderRadius=3
        ))

    def _register_fonts(self):
        """Register custom fonts for PDF generation."""
        # Add custom fonts here if needed
        pass

    def _get_page_size(self, size: str):
        """Get the page size based on the selected option."""
        sizes = {
            "A4": A4,
            "Letter": LETTER,
            "Legal": LEGAL
        }
        return sizes.get(size, A4)

    def _create_title_style(self):
        """Create a custom style for titles."""
        return ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            textColor=colors.HexColor('#2C3E50')
        )

    def _create_heading_style(self):
        """Create a custom style for headings."""
        return ParagraphStyle(
            'CustomHeading',
            parent=self.styles['Heading2'],
            fontSize=18,
            spaceAfter=20,
            textColor=colors.HexColor('#34495E')
        )

    def _create_metadata_section(self, doc: SimpleDocTemplate, metadata: Dict[str, Any]):
        """Create the metadata section of the document."""
        elements = []
        
        # Title
        title_style = self._create_title_style()
        elements.append(Paragraph(metadata.get("title", "Untitled"), title_style))
        elements.append(Spacer(1, 0.5 * inch))
        
        # Metadata table
        if metadata.get("include_metadata", True):
            metadata_data = [
                ["Author", metadata.get("author", os.getenv("PROF_NAME", "Unknown"))],
                ["Created", metadata.get("created_at", datetime.now().strftime("%Y-%m-%d"))],
                ["Course", metadata.get("course", "Unknown")]
            ]
            
            metadata_table = Table(metadata_data, colWidths=[2*inch, 4*inch])
            metadata_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#ECF0F1')),
                ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#2C3E50')),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#BDC3C7'))
            ]))
            elements.append(metadata_table)
            elements.append(Spacer(1, 0.5 * inch))
            
        return elements

    def _create_toc(self, doc: SimpleDocTemplate, sections: List[str]):
        """Create a table of contents."""
        elements = []
        
        heading_style = self._create_heading_style()
        elements.append(Paragraph("Table of Contents", heading_style))
        elements.append(Spacer(1, 0.25 * inch))
        
        for i, section in enumerate(sections, 1):
            elements.append(Paragraph(f"{i}. {section}", self.styles["Normal"]))
            
        elements.append(Spacer(1, 0.5 * inch))
        return elements

    def _create_code_block(self, code: str, language: str = "python"):
        """Create a formatted code block."""
        # Add syntax highlighting based on language
        return Paragraph(f"<pre>{code}</pre>", self.styles["Code"])

    def _create_header(self, title: str, course_name: str) -> List[Any]:
        """Create the header section of the document."""
        elements = []
        
        # Title
        elements.append(Paragraph(title, self.styles['Heading1']))
        elements.append(Spacer(1, 0.2 * inch))
        
        # Course and date
        elements.append(Paragraph(f"Course: {course_name}", self.styles['BodyText']))
        elements.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", self.styles['BodyText']))
        elements.append(Spacer(1, 0.5 * inch))
        
        return elements

    def _create_deliverables_section(self, deliverables: List[Deliverable]) -> List[Any]:
        """Create the deliverables section."""
        elements = []
        elements.append(Paragraph("Deliverables", self.styles['Heading2']))
        elements.append(Spacer(1, 0.2 * inch))
        
        for i, deliverable in enumerate(deliverables, 1):
            elements.append(Paragraph(f"{i}. {deliverable.name}", self.styles['Heading3']))
            elements.append(Paragraph(deliverable.description, self.styles['BodyText']))
            elements.append(Paragraph(f"Format: {deliverable.format}", self.styles['BodyText']))
            if deliverable.deadline:
                elements.append(Paragraph(f"Deadline: {deliverable.deadline}", self.styles['BodyText']))
            elements.append(Spacer(1, 0.2 * inch))
        
        return elements

    def _create_constraints_section(self, constraints: List[TechnicalConstraint]) -> List[Any]:
        """Create the technical constraints section."""
        elements = []
        elements.append(Paragraph("Technical Constraints", self.styles['Heading2']))
        elements.append(Spacer(1, 0.2 * inch))
        
        data = [["Constraint", "Description", "Required"]]
        for constraint in constraints:
            data.append([
                constraint.name,
                constraint.description,
                "Yes" if constraint.required else "No"
            ])
        
        table = Table(data, colWidths=[2*inch, 4*inch, 1*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#E9ECEF')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#212529')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#FFFFFF')),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.HexColor('#495057')),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#DEE2E6')),
            ('ALIGN', (2, 1), (2, -1), 'CENTER'),
        ]))
        
        elements.append(table)
        elements.append(Spacer(1, 0.3 * inch))
        
        return elements

    def _create_performance_section(self, requirements: List[PerformanceRequirement]) -> List[Any]:
        """Create the performance requirements section."""
        elements = []
        elements.append(Paragraph("Performance Requirements", self.styles['Heading2']))
        elements.append(Spacer(1, 0.2 * inch))
        
        data = [["Metric", "Target", "Description"]]
        for req in requirements:
            data.append([req.metric, req.target, req.description])
        
        table = Table(data, colWidths=[2*inch, 2*inch, 3*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#E9ECEF')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#212529')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#FFFFFF')),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.HexColor('#495057')),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#DEE2E6')),
        ]))
        
        elements.append(table)
        elements.append(Spacer(1, 0.3 * inch))
        
        return elements

    def generate_assignment_pdf(self, assignment_data: Dict[str, Any], output_path: str, options: Dict[str, Any]):
        """Generate a PDF for an assignment."""
        print(f"\n[PDFGenerator] Starting assignment PDF generation for: {output_path}")
        print(f"[PDFGenerator] Assignment data keys: {list(assignment_data.keys())}")
        print(f"[PDFGenerator] Options: {options}")
        
        try:
            # Set up the document
            print("[PDFGenerator] Setting up document template")
            doc = SimpleDocTemplate(
                output_path,
                pagesize=self._get_page_size(options.get("paper_size", "A4")),
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=72
            )
            
            elements = []
            
            # Add metadata
            print("[PDFGenerator] Adding metadata section")
            metadata = {
                "title": assignment_data.get("title", "Assignment"),
                "course": assignment_data.get("course_name", "Unknown Course"),
                "created_at": datetime.now().strftime("%Y-%m-%d"),
                "include_metadata": options.get("include_metadata", True)
            }
            elements.extend(self._create_metadata_section(doc, metadata))
            
            # Add table of contents if requested
            if options.get("include_toc", True):
                print("[PDFGenerator] Adding table of contents")
                sections = ["Overview", "Problems", "Instructions"]
                elements.extend(self._create_toc(doc, sections))
            
            # Add assignment overview
            print("[PDFGenerator] Adding assignment overview")
            heading_style = self._create_heading_style()
            elements.append(Paragraph("Assignment Overview", heading_style))
            elements.append(Paragraph(assignment_data.get("description", ""), self.styles["Normal"]))
            elements.append(Spacer(1, 0.5 * inch))
            
            # Add problems
            if "problems" in assignment_data:
                print("[PDFGenerator] Adding problems section")
                elements.append(Paragraph("Problems", heading_style))
                for i, problem in enumerate(assignment_data["problems"], 1):
                    # Add page break before long essay problems
                    if problem.get("type") == "long_essay":
                        elements.append(PageBreak())
                    
                    # Problem header with beautified type
                    problem_type = problem.get("type", "Problem").replace("_", " ").title()
                    elements.append(Paragraph(f"Problem {i} ({problem_type})", self.styles["Heading3"]))
                    
                    # Problem description with beautified text
                    description = problem.get("description", "").replace("_", " ")
                    elements.append(Paragraph(description, self.styles["Normal"]))
                    
                    # Add options for multiple choice problems
                    if problem.get("type") == "multiple_choice" and "options" in problem:
                        for option in problem["options"]:
                            elements.append(Paragraph(f"• {option}", self.styles["Normal"]))
                        
                        # Add answer for MCQ problems
                        if "answer" in problem:
                            elements.append(Spacer(1, 0.2 * inch))
                            elements.append(Paragraph("Answer:", self.styles["Heading4"]))
                            elements.append(Paragraph(problem["answer"], self.styles["Normal"]))
                    
                    # Add hints if available
                    if "hints" in problem:
                        elements.append(Paragraph("Hints:", self.styles["Heading4"]))
                        for hint in problem["hints"]:
                            elements.append(Paragraph(f"• {hint}", self.styles["Normal"]))
                    
                    # Add test cases if available
                    if "test_cases" in problem:
                        elements.append(Paragraph("Test Cases:", self.styles["Heading4"]))
                        for tc in problem["test_cases"]:
                            elements.append(Paragraph(f"Input: {tc['input']}", self.styles["Normal"]))
                            elements.append(Paragraph(f"Expected Output: {tc['output']}", self.styles["Normal"]))
                    
                    # Add documentation requirements if available
                    if "documentation_requirements" in problem:
                        elements.append(Paragraph("Documentation Requirements:", self.styles["Heading4"]))
                        for req in problem["documentation_requirements"]:
                            elements.append(Paragraph(f"• {req}", self.styles["Normal"]))
                    
                    # Add page break after long essay problems
                    if problem.get("type") == "long_essay":
                        elements.append(PageBreak())
                    # Add spacer for short essay problems to ensure two per page
                    elif problem.get("type") == "short_essay":
                        elements.append(Spacer(1, 0.3 * inch))
                    else:
                        elements.append(Spacer(1, 0.3 * inch))
            
            # Add general instructions
            if "header" in assignment_data and "instructions" in assignment_data["header"]:
                print("[PDFGenerator] Adding instructions section")
                elements.append(Paragraph("Instructions", heading_style))
                for instruction in assignment_data["header"]["instructions"]:
                    elements.append(Paragraph(f"• {instruction}", self.styles["Normal"]))
                elements.append(Spacer(1, 0.5 * inch))
            
            # Build the PDF
            print("[PDFGenerator] Building PDF")
            doc.build(elements)
            print(f"[PDFGenerator] PDF generated successfully at: {output_path}")
            
        except Exception as e:
            print(f"[PDFGenerator] ERROR: Failed to generate assignment PDF: {str(e)}")
            raise

    def generate_project_pdf(self, project_data: Dict[str, Any], output_path: str, options: Dict[str, Any]):
        """Generate a PDF for a project."""
        print(f"\n[PDFGenerator] Starting project PDF generation for: {output_path}")
        print(f"[PDFGenerator] Project data keys: {list(project_data.keys())}")
        print(f"[PDFGenerator] Options: {options}")
        
        try:
            # Set up the document
            print("[PDFGenerator] Setting up document template")
            doc = SimpleDocTemplate(
                output_path,
                pagesize=self._get_page_size(options.get("paper_size", "A4")),
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=72
            )
            
            elements = []
            
            # Add metadata
            print("[PDFGenerator] Adding metadata section")
            metadata = {
                "title": project_data.get("title", "Project"),
                "course": project_data.get("course_name", "Unknown Course"),
                "created_at": datetime.now().strftime("%Y-%m-%d"),
                "version": "1.0",
                "include_metadata": options.get("include_metadata", True)
            }
            elements.extend(self._create_metadata_section(doc, metadata))
            
            # Add table of contents if requested
            if options.get("include_toc", True):
                print("[PDFGenerator] Adding table of contents")
                sections = ["Overview", "Requirements", "Deliverables", "Timeline", "Technical Details"]
                elements.extend(self._create_toc(doc, sections))
                
            # Add project overview
            print("[PDFGenerator] Adding project overview")
            heading_style = self._create_heading_style()
            elements.append(Paragraph("Project Overview", heading_style))
            elements.append(Paragraph(project_data.get("description", ""), self.styles["Normal"]))
            elements.append(Spacer(1, 0.5 * inch))
            
            # Add requirements
            if "requirements" in project_data:
                print("[PDFGenerator] Adding requirements section")
                elements.append(Paragraph("Requirements", heading_style))
                for req in project_data["requirements"]:
                    elements.append(Paragraph(f"• {req}", self.styles["Normal"]))
                elements.append(Spacer(1, 0.5 * inch))
            
            # Add deliverables
            if "deliverables" in project_data:
                print("[PDFGenerator] Adding deliverables section")
                elements.append(Paragraph("Deliverables", heading_style))
                for deliverable in project_data["deliverables"]:
                    elements.append(Paragraph(f"• {deliverable}", self.styles["Normal"]))
                elements.append(Spacer(1, 0.5 * inch))
            
            # Add timeline
            if "timeline" in project_data:
                print("[PDFGenerator] Adding timeline section")
                elements.append(Paragraph("Timeline", heading_style))
                for milestone in project_data["timeline"]:
                    elements.append(Paragraph(f"• {milestone}", self.styles["Normal"]))
                elements.append(Spacer(1, 0.5 * inch))
            
            # Add technical details
            if "technical_details" in project_data:
                print("[PDFGenerator] Adding technical details section")
                elements.append(Paragraph("Technical Details", heading_style))
                for detail in project_data["technical_details"]:
                    elements.append(Paragraph(f"• {detail}", self.styles["Normal"]))
                elements.append(Spacer(1, 0.5 * inch))
            
            # Build the PDF
            print("[PDFGenerator] Building PDF")
            doc.build(elements)
            print(f"[PDFGenerator] PDF generated successfully at: {output_path}")
            
        except Exception as e:
            print(f"[PDFGenerator] ERROR: Failed to generate project PDF: {str(e)}")
            raise

    def generate_exam_pdf(self, exam_data: Dict[str, Any], output_path: str, options: Dict[str, Any]):
        """Generate a PDF for an exam."""
        print(f"\n[PDFGenerator] Starting exam PDF generation for: {output_path}")
        print(f"[PDFGenerator] Exam data keys: {list(exam_data.keys())}")
        print(f"[PDFGenerator] Options: {options}")
        
        try:
            # Set up the document
            print("[PDFGenerator] Setting up document template")
            doc = SimpleDocTemplate(
                output_path,
                pagesize=self._get_page_size(options.get("paper_size", "A4")),
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=72
            )
            
            elements = []
            
            # Add metadata
            print("[PDFGenerator] Adding metadata section")
            metadata = {
                "title": exam_data.get("title", "Exam"),
                "course": exam_data.get("course_name", "Unknown Course"),
                "created_at": datetime.now().strftime("%Y-%m-%d"),
                "include_metadata": options.get("include_metadata", True)
            }
            elements.extend(self._create_metadata_section(doc, metadata))
            
            # Add table of contents if requested
            if options.get("include_toc", True):
                print("[PDFGenerator] Adding table of contents")
                sections = ["Instructions", "Questions"]
                elements.extend(self._create_toc(doc, sections))
            
            # Add exam instructions
            print("[PDFGenerator] Adding instructions section")
            heading_style = self._create_heading_style()
            elements.append(Paragraph("Instructions", heading_style))
            if "instructions" in exam_data:
                for instruction in exam_data["instructions"]:
                    elements.append(Paragraph(f"• {instruction}", self.styles["Normal"]))
            elements.append(Spacer(1, 0.5 * inch))
            
            # Add questions
            if "questions" in exam_data:
                print("[PDFGenerator] Adding questions section")
                elements.append(Paragraph("Questions", heading_style))
                
                for i, question in enumerate(exam_data["questions"], 1):
                    # Add page break before long essay questions
                    if question.get("type") == "long_essay":
                        elements.append(PageBreak())
                    
                    # Question header with beautified type
                    question_type = question.get("type", "Unknown Type").replace("_", " ").title()
                    elements.append(Paragraph(f"Question {i} ({question_type})", self.styles["Heading3"]))
                    
                    # Question text with beautified text
                    question_text = question.get("question", "").replace("_", " ")
                    elements.append(Paragraph(question_text, self.styles["Normal"]))
                    
                    # Add options for multiple choice questions
                    if question.get("type") == "multiple_choice" and "options" in question:
                        for option in question["options"]:
                            elements.append(Paragraph(f"• {option}", self.styles["Normal"]))
                        
                        # Add answer for MCQ questions
                        if "answer" in question:
                            elements.append(Spacer(1, 0.2 * inch))
                            elements.append(Paragraph("Answer:", self.styles["Heading4"]))
                            elements.append(Paragraph(question["answer"], self.styles["Normal"]))
                    
                    # Add test cases for coding questions
                    if question.get("type") == "coding" and "test_cases" in question:
                        elements.append(Paragraph("Test Cases:", self.styles["Heading4"]))
                        for tc in question["test_cases"]:
                            elements.append(Paragraph(f"Input: {tc['input']}", self.styles["Normal"]))
                            elements.append(Paragraph(f"Expected Output: {tc['output']}", self.styles["Normal"]))
                    
                    # Add solution if available
                    if "solution" in question:
                        elements.append(Paragraph("Solution:", self.styles["Heading4"]))
                        elements.append(Paragraph(question["solution"], self.styles["Normal"]))
                    
                    # Add page break after long essay questions
                    if question.get("type") == "long_essay":
                        elements.append(PageBreak())
                    # Add spacer for short essay questions to ensure two per page
                    elif question.get("type") == "short_essay":
                        elements.append(Spacer(1, 0.3 * inch))
                    else:
                        elements.append(Spacer(1, 0.3 * inch))
            
            # Build the PDF
            print("[PDFGenerator] Building PDF")
            doc.build(elements)
            print(f"[PDFGenerator] PDF generated successfully at: {output_path}")
            
        except Exception as e:
            print(f"[PDFGenerator] ERROR: Failed to generate exam PDF: {str(e)}")
            raise

    def generate_rubric_pdf(self, rubric_data: Dict[str, Any], output_path: str, options: Dict[str, Any]):
        """Generate a PDF for a rubric."""
        print(f"\n[PDFGenerator] Starting rubric PDF generation for: {output_path}")
        print(f"[PDFGenerator] Rubric data keys: {list(rubric_data.keys())}")
        print(f"[PDFGenerator] Options: {options}")
        
        try:
            # Set up the document
            print("[PDFGenerator] Setting up document template")
            doc = SimpleDocTemplate(
                output_path,
                pagesize=self._get_page_size(options.get("paper_size", "A4")),
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=72
            )
            
            elements = []
            
            # Add metadata
            print("[PDFGenerator] Adding metadata section")
            metadata = {
                "title": rubric_data.get("title", "Rubric"),
                "created_at": datetime.now().strftime("%Y-%m-%d"),
                "include_metadata": options.get("include_metadata", True)
            }
            elements.extend(self._create_metadata_section(doc, metadata))
            
            # Add table of contents if requested
            if options.get("include_toc", True):
                print("[PDFGenerator] Adding table of contents")
                sections = ["Overview", "Criteria", "Scoring Levels"]
                elements.extend(self._create_toc(doc, sections))
            
            # Add rubric overview
            print("[PDFGenerator] Adding rubric overview")
            heading_style = self._create_heading_style()
            elements.append(Paragraph("Rubric Overview", heading_style))
            elements.append(Paragraph(rubric_data.get("description", ""), self.styles["Normal"]))
            elements.append(Paragraph(f"Total Points: {rubric_data.get('total_points', 0):.1f}", self.styles["Normal"]))
            elements.append(Spacer(1, 0.5 * inch))
            
            # Add criteria table
            print("[PDFGenerator] Adding criteria table")
            elements.append(Paragraph("Criteria", heading_style))
            
            # Create table data
            table_data = [["Criterion", "Weight", "Max Score", "Description"]]
            for criterion in rubric_data.get("criteria", []):
                table_data.append([
                    criterion["name"],
                    f"{criterion['weight']:.1f}%",
                    f"{criterion['max_score']:.1f}",
                    criterion["description"]
                ])
            
            # Create and style the table
            table = Table(table_data, colWidths=[2*inch, 1*inch, 1*inch, 3*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#E9ECEF')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#212529')),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#FFFFFF')),
                ('TEXTCOLOR', (0, 1), (-1, -1), colors.HexColor('#495057')),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#DEE2E6')),
                ('ALIGN', (1, 1), (2, -1), 'CENTER'),
                ('WORDWRAP', (0, 0), (-1, -1), True),  # Enable word wrapping for all cells
                ('LEFTPADDING', (0, 0), (-1, -1), 6),  # Add left padding
                ('RIGHTPADDING', (0, 0), (-1, -1), 6),  # Add right padding
                ('TOPPADDING', (0, 0), (-1, -1), 6),    # Add top padding
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),  # Add bottom padding
            ]))
            
            elements.append(table)
            elements.append(Spacer(1, 0.5 * inch))
            
            # Add scoring levels
            print("[PDFGenerator] Adding scoring levels")
            elements.append(Paragraph("Scoring Levels", heading_style))
            
            for criterion in rubric_data.get("criteria", []):
                elements.append(Paragraph(criterion["name"], self.styles["Heading3"]))
                for level in criterion.get("levels", []):
                    elements.append(Paragraph(
                        f"• {level['description']}: {level['feedback']}",
                        self.styles["Normal"]
                    ))
                elements.append(Spacer(1, 0.2 * inch))
            
            # Build the PDF
            print("[PDFGenerator] Building PDF")
            doc.build(elements)
            print(f"[PDFGenerator] PDF generated successfully at: {output_path}")
            
        except Exception as e:
            print(f"[PDFGenerator] ERROR: Failed to generate rubric PDF: {str(e)}")
            raise 