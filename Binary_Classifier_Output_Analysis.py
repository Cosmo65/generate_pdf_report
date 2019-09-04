# WARNING
# This code utilizes Pandas 0.25.x and will not work with earlier
# versions of Pandas since the ability to groupby aggregation with
# multiple lambdas was added on July 18, 2019
# Pandas can be upgraded using pip in the terminal (windows):
#   python -m pip install --upgrade pandas
# More info here: https://pandas.pydata.org/pandas-docs/stable/whatsnew/v0.25.0.html#groupby-aggregation-with-multiple-lambdas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.platypus import Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.rl_config import defaultPageSize
from reportlab.lib.units import inch
from reportlab.lib import colors

# Global variables used for formatting final report
styles = getSampleStyleSheet()  # Ex styles Normal,Title, Italic, Heading1


def main():
    '''This code returns a text and .pdf summary of the technical screening Qs.
    The confusion matrix, used to calculate sensitivity and specificity, are
    calculated using the grouping and aggregate function capabilties of the
    pandas dataframe.
    The ROC curve is obtained by calculating the True Positive Rate (sensitiv.)
    and False Positive Rate (1-specif.) for range of probability thresholds
    between 0 and 1. The resolution of the plot increases with the step-size,
    or resolution, of the calculation. A step size of 0.01 is adequate
    The area under the ROC curve was calculated using the trapezoid numerical
    integral method. This approximation improves wtih smaller step size, but
    was also noted to be adequate at a step size fo 0.01.'''

    model_outcome_df = get_df()   # store model output in a dataframe

    # unity_array is prob_threshold parameter sweep, [0,1]
    # Decreased step_size from 0.01 to 0.001 inc time from about 1->8 sec
    step_size = 0.01
    unity_array = np.arange(0, 1.005, step_size)

    # This list comprehension returns a list of tuples (false positive rate,
    #  true positive rate) for each threshold specified in the unity_array
    tuple_results = [
        calc_confusion_matrix(t, df)
        for t in unity_array
        for df in [model_outcome_df]]

    # The tuple results are unzipped and unpacked.
    # More information: https://docs.python.org/3/library/functions.html#zip
    false_positive_rate, sensitivity = list(zip(*tuple_results))
    AUC = round(calc_area_under_curve(false_positive_rate, sensitivity), 2)
    plt = plot_ROC_curve(false_positive_rate, sensitivity, unity_array, AUC)
    results_dict = calc_confusion_matrix(0.5, model_outcome_df, True)
    summary_text = (
    '''At a threshold of 0.5 the sensitivity is %s and the specificity is %s. \n
    The area under the curve was approximated to be %s using the trapezoid rule. \n
    The ROC curve was saved to the same directory as the .py file under the name 'JRD_ROC_Curve.jpeg'. \n
    See the full generated report under the name 'Joshua_Douglas_Technical_Screening.pdf' for more detail.
    ''' % (round(results_dict["Sensitivity"],3), round(results_dict["Specificity"],3), round(AUC,2)))
    print(summary_text)
    generate_report(
        plt, results_dict, AUC)

# The next section of code pulls the data and conducts the calculations
###############################################################################


def get_df():
    """Returns the provided model_outcome data in a Pandas dataframe"""
    url = 'https://raw.githubusercontent.com/aMetallurgist/generate_pdf_report/master/sample_data.txt'
    model_outcome_df = pd.read_csv(url, index_col=0)
    return model_outcome_df


def calc_confusion_matrix(threshold, model_outcome_df, return_full_results=False):
    """Returns results from the confusion matrix, with datatype dependend on the 'return_full_results' boolean

    Input:
        threshold - integer, [0,1], the predicted probability threshold,
                above which the value is assumed to be class 1
        model_outcome_df - a three column Pandas dataframe, index, class
                (0 or 1), predicted_probability [0,1]
        return_full_results - a boolean which determines output format.

    Return:
        1) return_full_results = False: returns the parameters required to
            calculate a Receiver Operating Characteristic (ROC) Curve in the
            tuple: (1 - model_specificity, model_sensitivity)
        Note:
            1 - model_specificity is referred to as the False Positive Rate,
            the X-axis on the ROC Curve

        2) return_full_results = True: return full confusion matrix
    """
    # The following two lambda functions are aggregate functions which count
    # the number of items below / equal to or greater than the threshold
    # lambda func: https://docs.python.org/3/reference/expressions.html#lambda
    # agg func: https://pandas.pydata.org/pandas-docs/stable/user_guide/groupby.html#aggregation
    def count_pred_class_zero(x): return sum(list(x < threshold))
    def count_pred_class_one(x): return sum(list(x >= threshold))

    # Pandas dataframes allow for complex grouping and calculations without
    # looping. The .groupby("class") method breaks the dataframe into two
    # groups - class 0 and class 1. The .predicted_prob method focuses the
    # aggregate calculations on the predicted_prob column. The .agg method
    # applies aggregate function(s) on the groups (class 0 and class 1)
    # In this case a list of aggregate functions was passed through to allow
    # multiple aggregations on one column. The resulting dataframe is a 2 x 2
    # confusion matrix grouped by known and predicted class.
    # https://pandas.pydata.org/pandas-docs/stable/whatsnew/v0.25.0.html?highlight=multiple%20lambda#groupby-aggregation-with-multiple-lambdas
    confusion_matrix_df = model_outcome_df.groupby("class").predicted_prob.agg(
        [count_pred_class_zero, count_pred_class_one])

    # the next two lines simply change the column and row labels, respectively
    confusion_matrix_df.columns = ['Predicted_Class_Zero', 'Predicted_Class_One']
    confusion_matrix_df.index = ['Known_Class_Zero', 'Known_Class_One']

    # results_dictionary holds the confusion matrix counts,
    # along with sensitivity and specificity calculations
    results_dict = {'True_Neg': confusion_matrix_df['Predicted_Class_Zero']['Known_Class_Zero'],
                    'False_Pos': confusion_matrix_df['Predicted_Class_One']['Known_Class_Zero'],
                    'False_Neg': confusion_matrix_df['Predicted_Class_Zero']['Known_Class_One'],
                    'True_Pos': confusion_matrix_df['Predicted_Class_One']['Known_Class_One']}
    results_dict['Sensitivity'] = results_dict['True_Pos'] / (results_dict['True_Pos']+results_dict['False_Neg'])
    results_dict['Specificity'] = results_dict['True_Neg'] / (results_dict['True_Neg']+results_dict['False_Pos'])

    # if return_full_results is true then return full confusion_matrix summary
    if return_full_results:
        return results_dict
    else:       # otherwise the (y,x) values for the ROC curve are returned
        return (1-results_dict['Specificity'], results_dict['Sensitivity'])


def calc_area_under_curve(xdata, ydata):
    '''Return area under curve with lists of the x-coord and y-coord data.
    This function approximates a definite integral using the trapezoid rule,
    so the approximation will be more accurate as deltaX approaches zero.'''
    area = 0
    for i in range(len(xdata)-1):
        # area of a trapezoid = 0.5*deltaX*(y1+y2)
        area += 0.5*((abs(xdata[i+1]-xdata[i]))*(ydata[i+1]+ydata[i]))
    return area

# The next section of code visualizes the ROC plot
###############################################################################


def plot_ROC_curve(false_positive_rate, sensitivity, unity_array, AUC):
    '''Return plot of ROC curve with lists of false_pos_rate and sensitivity'''
    z = zip(false_positive_rate, sensitivity)
    z = list(z)
    fig, ax = plt.subplots()
    ax.plot(false_positive_rate, sensitivity, color='black',
            label='Area Under ROC Curve = %s' % AUC)
    ax.plot(unity_array, unity_array, color='red', linestyle='--', label='Random Guess (AUC = 0.5)')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.005)
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=14)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=14)
    plt.title('Receiver Operating Characteristic Curve', fontsize=16)
    ax.legend(loc='lower right')
    # find example threshold values to create annotations
    i1 = max(np.where(unity_array <= 0.3501)[0])
    coord1 = (round(false_positive_rate[i1],3), round(sensitivity[i1], 3))
    i2 = max(np.where(unity_array <= 0.6001)[0])
    coord2 = (round(false_positive_rate[i2],3), round(sensitivity[i2], 3))
    plt.annotate('Threshold = 0.35 \n(FPR = %s, TPR = %s)' % (coord1[0], coord1[1]),
                 xy=(coord1[0], coord1[1]), xytext=(0.41,0.81), arrowprops=dict(facecolor='black', shrink=0.03))
    plt.annotate('Threshold = 0.6 \n(FPR = %s, TPR = %s)' % (coord2[0], coord2[1]),
                 xy=(coord2[0], coord2[1]), xytext=(0.15,0.54), arrowprops=dict(facecolor='black', shrink=0.03))
    plt.savefig("JRD_ROC_Curve.jpeg", dpi=600, aspect='equal')
    plt.close()

# The next section is purely formatting, used to generate the pdf report
###############################################################################

def generate_report(plot_object, results_dict, AUC):
    '''Use the simple document template within the reportlab library to
    generate a pdf report of the results. The story object is a ordered
    collection of custom 'Flowables' objects such as paragraphs, images,
    spacers, and tables.'''
    PAGE_WIDTH = defaultPageSize[0]
    doc = SimpleDocTemplate("Model_Analysis.pdf")
    Story = [Spacer(1, 1.0*inch)]  # add a blank space
    style = styles["Normal"]       # use template style, "Normal"
    for line in document_text(results_dict, AUC):
        if line[0] == "Q" or line[0] == "A":
            p = Paragraph(line, style)
            Story.append(p)             # add a paragraph with text 'line'
        elif line == "Create Table":
            data_model_output = [    # actual confusion matrix
                [" ", "Predicted Class Zero", "Predicted Class One"],
                ["Known Class Zero", results_dict["True_Neg"], results_dict["False_Pos"]],
                ["Known Class One", results_dict["False_Neg"], results_dict["True_Pos"]]]
            t1 = Table(data_model_output)
            t1.setStyle(TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                                    ('GRID', (1, 0), (2, 0), 0.25, colors.black),
                                    ('GRID', (0, 1), (-1, -1), 0.25, colors.black)]))
            styles["Heading5"].alignment = 1  # Override Headings5 stand format to center align
            p = Paragraph("Binary Classification Model Confusion Matrix Results",
                          styles["Heading5"])
            Story.append(p)
            Story.append(t1)     # add a table to display data_model_output
            data_example_confusion_matrix = [    # sample confusion matrix
                ["", "Predicted Condition Negative", "Predicted Condition Positive"],
                ["Known Condition Negative", "True Negative", "False Positive"],
                ["Known Condition Postiive", "False Negative", "True Positive"]]
            t2 = Table(data_example_confusion_matrix)
            t2.setStyle(TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                                    ('GRID', (1, 0), (2, 0), 0.25, colors.black),
                                    ('GRID', (0, 1), (-1, -1), 0.25, colors.black)]))
            Story.append(Spacer(1, 0.2*inch))
            p = Paragraph("Confusion Matrix Interpretation", styles["Heading5"])
            Story.append(p)
            Story.append(t2)
        Story.append(Spacer(1, 0.2*inch))

    f = open("JRD_ROC_Curve.jpeg", 'rb')
    Story.append(          # add ROC plot
        Image(f, width=PAGE_WIDTH*0.7, height=PAGE_WIDTH*0.5, hAlign='CENTER'))
    doc.build(Story, onFirstPage=myFirstPage, onLaterPages=myLaterPages)


def myFirstPage(canvas, doc):
    '''Output title, subtitle, and footer to first page'''
    PAGE_HEIGHT = defaultPageSize[1]
    PAGE_WIDTH = defaultPageSize[0]
    Title = "Joshua Douglas"
    Subtitle = "Binary Classifier Model ROC Analysis"
    pageinfo = "Using Generic Data"

    canvas.saveState()
    canvas.setFont('Times-Bold', 16)
    canvas.drawCentredString(PAGE_WIDTH/2.0, PAGE_HEIGHT-98, Title)
    canvas.setFont('Times-Roman', 16)
    canvas.drawCentredString(PAGE_WIDTH/2.0, PAGE_HEIGHT-118, Subtitle)
    canvas.setFont('Times-Roman', 9)
    canvas.drawString(inch, 0.75 * inch, "First Page / %s" % pageinfo)
    canvas.restoreState()


def myLaterPages(canvas, doc):
    '''Output footer to each page after the first page'''
    pageinfo = "Binary Classifier Model ROC Analysis"
    canvas.saveState()
    canvas.setFont('Times-Roman', 9)
    canvas.drawString(inch, 0.75 * inch, "Page %d / %s" % (doc.page, pageinfo))
    canvas.restoreState()


def document_text(results_dict, AUC):
    '''This block of text is entered into the .pdf report within the
    generate_report function. '''
    text_lst = [
        "Q1. <i>Manually calculate the sensitivity and specificity of the model, using a predicted_prob threshold of greater than or equal to 0.5.</i>",
        ('''A1. The sensitivity and specificity can be calculated from the confusion matrix, provided below along
        with an overview of the confusion matrix interpretation. The sensitivity is defined as the
        true positive rate (True Positive / (True Postive + False Negative)) and the specificity
        is defined as the true negative rate (True Negative / (True Negative + False Positive)). These ratios
        can be easily calculated with the provided matrix, yielding a <b> sensitivity value of %s and a
        specificity value of %s.</b>''' % (round(results_dict['Sensitivity'], 3), round(results_dict['Specificity'], 3))),
        "Create Table",
        "Q2. <i>Manually calculate the Area Under the Receiver Operating Characteristic Curve.</i>",
        ('''A2. The Receiver Operating Characteristic (ROC) Curve is a plot of true positive rate (TPR) against the
        false positive rate (FPR), across the range of model threshold settings. In our case the ROC plot data can be
        obtained by computing the confusion matrix, and corresponding TPR/FPR values, for probability threshold
        values between zero and one. The definite integral, or true area under the curve, can be approximated
        using numerical methods such as the Riemann Sum or Trapezoid rule. These approximations approach the true
        value as the resolution of the calculation increases, and both methods were found to converge to similar
        values when the parameter sweep step-size was set to 0.01. Using the trapezoid rule
        <b>the Area Under the Curve is approximately %s</b>.''' % (AUC)),
        "Q3. <i>Visualize the Receiver Operating Characterstic Curve.</i>",
        '''A3. The ROC curve is provided on the next page. The curve demonstrates that for all reasonable threshold values
        the proportion of correctly classified class 1 samples is greater than the proportion of the
        incorrectly classified class 0 samples (i.e. the model is better than random chance at all thresholds).
        With a AUC of %s the model is a reasonably good classifier, depending on the context. The ROC curve can be
        used to set the appropriate threshold, depending on the desired ratio of FPR to TPR. A review of the data
        indicates that a threshold of 0.35 (FPR = 0.466, TPR = 0.966) is appropriate if it is a priority to correctly classify
        true positives and a threshold of 0.6 (FPR = 0.070, TPR = 0.652) is appropriate if it is a priority to
        minimize the number of incorrectly classified negative / class zero events.''' % (AUC),
    ]
    return text_lst


if __name__ == "__main__":
    main()
