"""Step 1 - Data prep for the Legal Document Review Agent.

Generates a small corpus of synthetic, public-domain-style commercial legal
contracts as PDFs and stores them locally in ``legal_dummy_contracts/``. This is
the free, offline, reproducible stand-in for "download 10-20 public-domain legal
contract PDFs from CommonCrawl / OpenLegalData / GOV.UK and store them locally" -
no network or cloud blob storage required.

Each contract contains the full set of clauses a reviewer cares about
(indemnification, force majeure, limitation of liability, termination,
confidentiality, intellectual property, governing law, arbitration, data
protection, ...) with the specifics varied per document, so the hybrid retriever
and cross-encoder re-ranker have meaningful, distinguishable text to find.

You can also drop real downloaded contract PDFs into ``legal_dummy_contracts/``;
the rest of the pipeline treats every PDF in that folder identically.

Run:
    python generate_contracts.py
"""
from __future__ import annotations

import random
from datetime import datetime, timedelta

from fpdf import FPDF
from fpdf.enums import XPos, YPos

import config

CLIENTS = [
    "Contoso Corporation", "Fabrikam Inc.", "Adventure Works Ltd.",
    "Woodgrove Bank plc", "Northwind Traders LLC", "Tailspin Toys Inc.",
    "Alpine Ski House GmbH", "Coho Winery LLC", "Litware Inc.",
    "Proseware Systems Ltd.", "Wingtip Toys Inc.", "Margies Travel LLC",
    "Lucerne Publishing Ltd.", "Trey Research Inc.", "Fourth Coffee Company",
]
PROVIDERS = [
    "LegalTech AI Solutions", "Summit Advisory Group",
    "Blue Yonder Consulting", "Vector Professional Services",
]
AGREEMENT_TYPES = [
    "Master Services Agreement", "Software License and Services Agreement",
    "Professional Services Agreement", "Supply and Distribution Agreement",
    "Consulting Services Agreement",
]
NOTICE_PERIODS = [30, 60, 90]
LIABILITY_CAPS = [
    "USD 50,000",
    "USD 100,000",
    "USD 250,000",
    "USD 1,000,000",
    "the total fees paid by the Client in the twelve (12) months preceding the claim",
]
JURISDICTIONS = [
    ("the State of Delaware, United States", "the state and federal courts located in Wilmington, Delaware"),
    ("England and Wales", "the courts of London, England"),
    ("the State of California, United States", "the courts located in San Francisco, California"),
    ("the Republic of Ireland", "the courts of Dublin, Ireland"),
    ("the State of New York, United States", "the courts located in New York County, New York"),
]
ARBITRATION = [
    ("the American Arbitration Association (AAA)", "New York, New York"),
    ("the London Court of International Arbitration (LCIA)", "London, England"),
    ("the International Chamber of Commerce (ICC)", "Paris, France"),
    ("JAMS", "San Francisco, California"),
]
DATA_REGIMES = [
    "the EU General Data Protection Regulation (GDPR)",
    "the California Consumer Privacy Act (CCPA)",
    "the UK Data Protection Act 2018",
    "all applicable data protection and privacy laws",
]
FORCE_MAJEURE_EVENTS = [
    "acts of God, flood, fire, earthquake or explosion",
    "epidemic, pandemic or public health emergency",
    "war, invasion, hostilities or terrorist threat",
    "government order, embargo, or change in applicable law",
    "failure of public utilities, telecommunications, or the internet",
    "labor disputes, strikes, or industrial action",
]


def build_sections(idx: int) -> tuple[str, str, list[tuple[str, str]]]:
    """Return (title, filename_stub, [(heading, body), ...]) for one contract."""
    client = random.choice(CLIENTS)
    provider = random.choice(PROVIDERS)
    agreement = random.choice(AGREEMENT_TYPES)
    notice = random.choice(NOTICE_PERIODS)
    cap = random.choice(LIABILITY_CAPS)
    law, venue = random.choice(JURISDICTIONS)
    arb_body, arb_seat = random.choice(ARBITRATION)
    regime = random.choice(DATA_REGIMES)
    fm_events = ", ".join(random.sample(FORCE_MAJEURE_EVENTS, k=3))
    fee = random.choice([25000, 48000, 75000, 120000, 240000])
    survival = random.choice([2, 3, 5])
    warranty_days = random.choice([30, 60, 90])
    nonsolicit = random.choice([6, 12, 18, 24])

    start_date = datetime.now() - timedelta(days=random.randint(15, 540))
    start_str = start_date.strftime("%B %d, %Y")

    title = f"{agreement.upper()}"
    intro = (
        f"This {agreement} (the \"Agreement\") is made and entered into effective "
        f"as of {start_str} (the \"Effective Date\"), by and between {provider}, a "
        f"company providing professional and technology services (the \"Provider\"), "
        f"and {client} (the \"Client\"). The Provider and the Client may each be "
        f"referred to as a \"Party\" and collectively as the \"Parties\"."
    )

    sections: list[tuple[str, str]] = [
        ("RECITALS", intro),
        ("1. SERVICES AND SCOPE OF WORK",
         f"The Provider shall perform the services described in one or more "
         f"statements of work agreed by the Parties (the \"Services\"). The "
         f"Provider shall perform the Services in a professional and workmanlike "
         f"manner consistent with generally accepted industry standards. Any change "
         f"to the scope of the Services must be documented in a written change order "
         f"signed by both Parties."),
        ("2. PAYMENT TERMS AND FEES",
         f"In consideration for the Services, the Client shall pay the Provider fees "
         f"of USD {fee:,} payable in accordance with the applicable statement of "
         f"work. Unless otherwise stated, invoices are due within thirty (30) days "
         f"of receipt. Undisputed amounts not paid when due shall accrue interest at "
         f"the rate of 1.5% per month or the maximum rate permitted by law, "
         f"whichever is lower."),
        ("3. TERM AND TERMINATION",
         f"This Agreement commences on the Effective Date and continues until the "
         f"Services are completed, unless terminated earlier. Either Party may "
         f"terminate this Agreement for convenience by providing not less than "
         f"{notice} days prior written notice to the other Party. Either Party may "
         f"terminate immediately if the other Party commits a material breach that "
         f"remains uncured for fifteen (15) days after written notice. Upon "
         f"termination, the Client shall pay for all Services performed up to the "
         f"effective date of termination."),
        ("4. LIMITATION OF LIABILITY",
         f"Except for liability arising from a breach of confidentiality, the "
         f"indemnification obligations, or a Party's gross negligence or willful "
         f"misconduct, each Party's aggregate liability arising out of or related to "
         f"this Agreement shall not exceed {cap}. In no event shall either Party be "
         f"liable for any indirect, incidental, special, consequential, or punitive "
         f"damages, including lost profits, even if advised of the possibility of "
         f"such damages."),
        ("5. INDEMNIFICATION",
         f"The Provider shall indemnify, defend, and hold harmless the Client and "
         f"its officers, directors, and employees from and against any third-party "
         f"claims, losses, liabilities, damages, and reasonable expenses (including "
         f"reasonable attorneys' fees) arising out of (a) the Provider's breach of "
         f"this Agreement, or (b) any allegation that the deliverables infringe or "
         f"misappropriate any third-party intellectual property right. The "
         f"indemnified Party shall provide prompt notice of any claim and reasonable "
         f"cooperation in the defense. This indemnity does not apply to the extent a "
         f"claim arises from the indemnified Party's own negligence."),
        ("6. FORCE MAJEURE",
         f"Neither Party shall be liable for any failure or delay in performing its "
         f"obligations (except payment obligations) to the extent such failure or "
         f"delay is caused by events beyond its reasonable control, including {fm_events}. "
         f"The affected Party shall give prompt written notice and use commercially "
         f"reasonable efforts to resume performance. If a force majeure event "
         f"continues for more than sixty (60) consecutive days, either Party may "
         f"terminate this Agreement upon written notice without liability."),
        ("7. CONFIDENTIALITY",
         f"Each Party agrees to protect the other Party's Confidential Information "
         f"using the same degree of care it uses for its own confidential "
         f"information, and in no event less than a reasonable degree of care, and "
         f"to use such information solely to perform this Agreement. The obligations "
         f"in this Section survive termination of this Agreement for a period of "
         f"{survival} years. Confidential Information does not include information "
         f"that is or becomes publicly available through no fault of the receiving "
         f"Party."),
        ("8. INTELLECTUAL PROPERTY",
         f"Each Party retains all right, title, and interest in its pre-existing "
         f"intellectual property. Upon full payment, the Provider assigns to the "
         f"Client all right, title, and interest in the deliverables created "
         f"specifically for the Client under this Agreement, excluding the "
         f"Provider's pre-existing materials and any general know-how, for which the "
         f"Provider grants the Client a non-exclusive, perpetual, royalty-free "
         f"license to the extent embedded in the deliverables."),
        ("9. WARRANTIES AND REPRESENTATIONS",
         f"The Provider warrants that the Services will materially conform to the "
         f"applicable statement of work for a period of {warranty_days} days after "
         f"delivery. As the Client's sole remedy for breach of this warranty, the "
         f"Provider shall re-perform the non-conforming Services at no additional "
         f"charge. Except as expressly stated, the Services are provided \"as is\" "
         f"and each Party disclaims all other warranties, whether express or "
         f"implied, including the implied warranties of merchantability and fitness "
         f"for a particular purpose."),
        ("10. DATA PROTECTION AND PRIVACY",
         f"To the extent the Provider processes personal data on behalf of the "
         f"Client, each Party shall comply with {regime}. The Provider shall "
         f"implement appropriate technical and organizational measures to protect "
         f"personal data against unauthorized access, loss, or disclosure, and shall "
         f"notify the Client without undue delay upon becoming aware of a personal "
         f"data breach."),
        ("11. GOVERNING LAW AND JURISDICTION",
         f"This Agreement shall be governed by and construed in accordance with the "
         f"laws of {law}, without regard to its conflict-of-laws principles. Subject "
         f"to Section 12, the Parties submit to the exclusive jurisdiction of {venue}."),
        ("12. DISPUTE RESOLUTION AND ARBITRATION",
         f"The Parties shall first attempt to resolve any dispute through good-faith "
         f"negotiation. Any dispute not resolved within thirty (30) days shall be "
         f"finally settled by binding arbitration administered by {arb_body} in "
         f"accordance with its rules. The seat of arbitration shall be {arb_seat}, "
         f"and the language of the arbitration shall be English. Judgment on the "
         f"award may be entered in any court of competent jurisdiction."),
        ("13. NON-SOLICITATION",
         f"During the term of this Agreement and for {nonsolicit} months thereafter, "
         f"neither Party shall directly solicit for employment any employee of the "
         f"other Party who was materially involved in the Services, except through "
         f"general advertising not specifically targeted at such employees."),
        ("14. ASSIGNMENT",
         f"Neither Party may assign this Agreement without the prior written consent "
         f"of the other Party, except that either Party may assign this Agreement to "
         f"a successor in connection with a merger, acquisition, or sale of "
         f"substantially all of its assets, upon written notice to the other Party."),
        ("15. MISCELLANEOUS AND ENTIRE AGREEMENT",
         f"This Agreement, together with its statements of work, constitutes the "
         f"entire agreement between the Parties and supersedes all prior or "
         f"contemporaneous agreements relating to its subject matter. No amendment "
         f"is effective unless in writing and signed by both Parties. If any "
         f"provision is held unenforceable, the remaining provisions remain in full "
         f"force and effect. The waiver of any breach shall not be deemed a waiver "
         f"of any subsequent breach.\n\n"
         f"IN WITNESS WHEREOF, the Parties have executed this Agreement as of the "
         f"Effective Date first written above."),
    ]

    stub = client.split()[0].replace(".", "").replace(",", "")
    return title, f"Contract_{idx:02d}_{stub}", sections


def render_pdf(title: str, sections: list[tuple[str, str]], path: str) -> None:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 14)
    pdf.multi_cell(0, 8, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(3)

    for heading, body in sections:
        pdf.set_font("Helvetica", "B", 11)
        pdf.multi_cell(0, 6, heading, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(0, 5.5, body, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(2)

    pdf.output(path)


def generate(num: int | None = None, seed: int = 42) -> int:
    """Generate ``num`` contract PDFs into the local corpus folder."""
    num = num or config.NUM_CONTRACTS
    out_dir = config.CONTRACTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # Start from a clean corpus so stale documents do not linger.
    for old in out_dir.glob("*.pdf"):
        old.unlink()

    random.seed(seed)
    for i in range(1, num + 1):
        title, stub, sections = build_sections(i)
        render_pdf(title, sections, str(out_dir / f"{stub}.pdf"))

    return num


def main() -> None:
    print(f"Generating {config.NUM_CONTRACTS} legal contracts...")
    count = generate()
    print(f"Success! {count} PDFs generated in '{config.CONTRACTS_DIR}'.")


if __name__ == "__main__":
    main()
